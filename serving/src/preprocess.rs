//! Image preprocessing that reproduces `anomaly_inspect.data.Preprocess` exactly.
//!
//! Decoding comes from the `image` crate. Resizing does not: `image::imageops::resize` uses
//! different rounding from Pillow, so this module ports Pillow's fixed-point convolution instead.
//! `tests/test_serving.py` compares the output bit for bit with Python on real images.

use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use serde::Deserialize;

const PRECISION_BITS: u32 = 32 - 8 - 2;
const BILINEAR_SUPPORT: f64 = 1.0;

#[derive(Debug, Clone, Deserialize)]
pub struct Preprocess {
    pub resize: u32,
    pub crop: u32,
    // Parsed as f64 and then narrowed, the same conversion path as Python's float to float32.
    pub mean: [f64; 3],
    pub std: [f64; 3],
}

impl Default for Preprocess {
    fn default() -> Self {
        Self {
            resize: 256,
            crop: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

/// The weights of one output pixel: where they start in the input, and the fixed-point values.
struct Kernel {
    start: usize,
    weights: Vec<i32>,
}

fn bilinear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Same arithmetic as Pillow's `precompute_coeffs` and `normalize_coeffs_8bpc`.
fn kernels(input: usize, output: usize) -> Vec<Kernel> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = BILINEAR_SUPPORT * filter_scale;
    let unit = f64::from(1u32 << PRECISION_BITS);

    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            // `as i64` truncates towards zero, like the C cast it replaces.
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as i64).min(input as i64) as usize;

            let raw: Vec<f64> = (start..end)
                .map(|x| bilinear((x as f64 - center + 0.5) / filter_scale))
                .collect();
            let total: f64 = raw.iter().sum();
            let weights = raw
                .iter()
                .map(|weight| {
                    let normalised = if total != 0.0 {
                        weight / total
                    } else {
                        *weight
                    };
                    let scaled = normalised * unit;
                    (if scaled >= 0.0 {
                        scaled + 0.5
                    } else {
                        scaled - 0.5
                    }) as i32
                })
                .collect();
            Kernel { start, weights }
        })
        .collect()
}

fn round_to_u8(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

fn rounding_offset() -> i32 {
    1 << (PRECISION_BITS - 1)
}

/// Bilinear resize of interleaved RGB: horizontal pass, then vertical. A pass is skipped if that
/// dimension does not change, as in Pillow.
pub fn resize_rgb(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (in_width, in_height) = (image.width() as usize, image.height() as usize);
    let (out_width, out_height) = (width as usize, height as usize);

    let horizontal: Vec<u8> = if in_width == out_width {
        image.as_raw().clone()
    } else {
        let table = kernels(in_width, out_width);
        let mut out = vec![0u8; out_width * in_height * 3];
        for (row_in, row_out) in image
            .as_raw()
            .chunks_exact(in_width * 3)
            .zip(out.chunks_exact_mut(out_width * 3))
        {
            for (kernel, pixel) in table.iter().zip(row_out.as_chunks_mut::<3>().0.iter_mut()) {
                let mut sums = [rounding_offset(); 3];
                for (offset, &weight) in kernel.weights.iter().enumerate() {
                    let source = (kernel.start + offset) * 3;
                    for channel in 0..3 {
                        sums[channel] += i32::from(row_in[source + channel]) * weight;
                    }
                }
                for channel in 0..3 {
                    pixel[channel] = round_to_u8(sums[channel]);
                }
            }
        }
        out
    };

    let vertical: Vec<u8> = if in_height == out_height {
        horizontal
    } else {
        let table = kernels(in_height, out_height);
        let stride = out_width * 3;
        let mut out = vec![0u8; stride * out_height];
        for (kernel, row_out) in table.iter().zip(out.chunks_exact_mut(stride)) {
            for (byte, value) in row_out.iter_mut().enumerate() {
                let mut sum = rounding_offset();
                for (offset, &weight) in kernel.weights.iter().enumerate() {
                    sum += i32::from(horizontal[(kernel.start + offset) * stride + byte]) * weight;
                }
                *value = round_to_u8(sum);
            }
        }
        out
    };

    RgbImage::from_raw(width, height, vertical).expect("buffer size follows from the dimensions")
}

impl Preprocess {
    /// Resize and centre crop. This is the image that the model sees, before normalisation.
    pub fn crop_image(&self, image: &DynamicImage) -> RgbImage {
        let resized = resize_rgb(&image.to_rgb8(), self.resize, self.resize);
        let start = (self.resize - self.crop) / 2;
        image::imageops::crop_imm(&resized, start, start, self.crop, self.crop).to_image()
    }

    /// Normalise to a float32 NCHW tensor with batch size 1.
    pub fn tensor(&self, cropped: &RgbImage) -> Array4<f32> {
        let side = self.crop as usize;
        let mean = self.mean.map(|value| value as f32);
        let std = self.std.map(|value| value as f32);
        let mut tensor = Array4::<f32>::zeros((1, 3, side, side));
        for (x, y, pixel) in cropped.enumerate_pixels() {
            for channel in 0..3 {
                let value = f32::from(pixel[channel]) / 255.0;
                tensor[[0, channel, y as usize, x as usize]] =
                    (value - mean[channel]) / std[channel];
            }
        }
        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn kernel_weights_sum_to_one() {
        let unit = 1i64 << PRECISION_BITS;
        for (input, output) in [(1024, 256), (700, 256), (100, 256), (256, 256)] {
            for kernel in kernels(input, output) {
                let total: i64 = kernel.weights.iter().map(|&w| i64::from(w)).sum();
                assert!(
                    (total - unit).abs() <= kernel.weights.len() as i64,
                    "total {total}"
                );
            }
        }
    }

    #[test]
    fn constant_image_stays_constant() {
        let image = RgbImage::from_pixel(300, 300, Rgb([17, 200, 91]));
        let resized = resize_rgb(&image, 256, 256);
        assert!(resized.pixels().all(|pixel| *pixel == Rgb([17, 200, 91])));
    }

    #[test]
    fn same_size_resize_is_identity() {
        let image = RgbImage::from_fn(8, 8, |x, y| Rgb([x as u8, y as u8, (x * y) as u8]));
        assert_eq!(resize_rgb(&image, 8, 8), image);
    }

    #[test]
    fn crop_and_normalisation_values() {
        let preprocess = Preprocess::default();
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(300, 300, Rgb([255, 0, 255])));
        let cropped = preprocess.crop_image(&image);
        assert_eq!(cropped.dimensions(), (224, 224));
        let tensor = preprocess.tensor(&cropped);
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
        assert!((tensor[[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[[0, 1, 100, 100]] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
    }

    #[test]
    fn grayscale_becomes_three_equal_channels() {
        let gray =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(64, 64, image::Luma([128])));
        let cropped = Preprocess::default().crop_image(&gray);
        assert!(cropped.pixels().all(|pixel| pixel.0 == [128, 128, 128]));
    }
}
