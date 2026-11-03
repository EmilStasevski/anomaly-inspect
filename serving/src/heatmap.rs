//! Heatmap rendering, using the same overlay as the Python report figures.

use std::io::Cursor;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::{ImageFormat, Rgb, RgbImage};
use ndarray::Array2;

const HEAT_COLOR: [f32; 3] = [229.0, 52.0, 11.0];
const MAX_OPACITY: f32 = 0.85;

/// Blend the score over the grayscale input. Scores below `low` are invisible, and scores at or
/// above `high` have full opacity. Both limits come from good calibration images.
pub fn render(input: &RgbImage, anomaly_map: &Array2<f32>, low: f32, high: f32) -> RgbImage {
    RgbImage::from_fn(input.width(), input.height(), |x, y| {
        let pixel = input.get_pixel(x, y);
        let gray =
            0.299 * f32::from(pixel[0]) + 0.587 * f32::from(pixel[1]) + 0.114 * f32::from(pixel[2]);
        let heat = ((anomaly_map[[y as usize, x as usize]] - low) / (high - low)).clamp(0.0, 1.0);
        let alpha = heat * MAX_OPACITY;
        Rgb(HEAT_COLOR.map(|color| ((1.0 - alpha) * gray + alpha * color).round() as u8))
    })
}

pub fn to_base64_png(image: &RgbImage) -> anyhow::Result<String> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(STANDARD.encode(bytes.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_scores_keep_the_grayscale_image() {
        let input = RgbImage::from_pixel(2, 2, Rgb([100, 100, 100]));
        let output = render(&input, &Array2::zeros((2, 2)), 1.0, 2.0);
        assert!(output.pixels().all(|pixel| pixel.0 == [100, 100, 100]));
    }

    #[test]
    fn high_scores_turn_towards_the_heat_colour() {
        let input = RgbImage::from_pixel(1, 1, Rgb([0, 0, 0]));
        let output = render(&input, &Array2::from_elem((1, 1), 9.0), 1.0, 2.0);
        assert_eq!(output.get_pixel(0, 0).0, [195, 44, 9]);
    }

    #[test]
    fn png_round_trip() {
        let input = RgbImage::from_pixel(3, 3, Rgb([1, 2, 3]));
        let bytes = STANDARD.decode(to_base64_png(&input).unwrap()).unwrap();
        assert_eq!(image::load_from_memory(&bytes).unwrap().to_rgb8(), input);
    }
}
