use dct_tiv::*;
use image::{imageops::FilterType, io::Reader as ImageReader};

fn main() {
    let palette = get_palette();
    let dct_matrices = create_calibration_matrices::<7, 16>(true);
    let matrices = create_calibration_matrices::<7, 16>(false);
    // println!("{}", matrices.len());
    let args = std::env::args().collect::<Vec<_>>();
    let width = 2;
    let height = 1;
    let img = ImageReader::open(&args[1])
        .unwrap()
        .decode()
        .unwrap()
        .resize_exact(7 * width, 16 * height, FilterType::Triangle)
        .to_rgb8();

    // let img_mat = Matrix::<8, 8>::from_image_part(&img, 10, 10);
    // println!("{}", img_mat);
    // let dct_mat = forward_dct(&img_mat);
    // let mut dct_mat = Matrix::<8, 8>::new();
    // dct_mat.set(3, 3, 1.0);
    // println!("{}", dct_mat);
    // let idct_mat = inverse_dct(&dct_mat);
    // println!("{}", idct_mat);

    // [2, 1, 1, 2, 3, 4, 6, 10,
    //  1, 1, 1, 2, 3, 4, 6, 8,
    //  1, 1, 1, 2, 2, 3, 6, 8,
    //  1, 1, 2, 2, 3, 5, 7, 9,
    //  1, 1, 2, 3, 4, 6, 8, 10,
    //  2, 2, 3, 4, 5, 7, 9, 12,
    //  2, 3, 4, 5, 7, 9, 11, 13,
    //  3, 3, 4, 6, 7, 9, 11, 14,
    //  3, 3, 4, 7, 9, 11, 13, 15,
    //  4
    //  4
    //  5
    //  6
    //  8
    //  10
    // 12]
    if args[2] == "--dct" {
        for line in textify_dct(&img, &dct_matrices, &palette) {
            println!(
                "{}{}{}",
                line.to_optimised_string(),
                termion::color::Reset.bg_str(),
                termion::color::Reset.fg_str()
            );
        }
    } else if args[2] == "--spatial" {
        // println!(
        //     "{}{}",
        //     termion::color::Reset.fg_str(),
        //     termion::color::Reset.bg_str()
        // );
        for line in textify_spatial(&img, &matrices, &palette) {
            println!(
                "{}{}{}",
                line.to_optimised_string(),
                termion::color::Reset.bg_str(),
                termion::color::Reset.fg_str()
            );
        }
    } else {
        eprintln!("expected --dct or --spatial");
    }
}
