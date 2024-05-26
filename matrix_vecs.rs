use fmtstring::Colour;
use image::io::Reader as ImageReader;
use std::collections::HashMap;
use std::f64::consts::PI;
use termion::color::Rgb;

fn print_calibration_image() -> Vec<char> {
    let broken_chars = "░▒▓◽◾▽△▷▻◁◅◈◌◍◔◬◭◮◸◹◺◿"; // chars with broken rendering (too big, or overlap other chars)
    let mut palette: Vec<char> = Vec::new();
    for y in 0..16 {
        for x in 0..16 {
            // if y == 15 && 12 < x && x < 15 {
            // continue;
            // }
            let ch = char::from_u32(0x2500 | (y << 4) | x).unwrap();
            if !broken_chars.contains(ch) {
                palette.push(ch);
            }
        }
    }
    for i in 32..127 {
        let ch = char::from_u32(i).unwrap();
        if !broken_chars.contains(ch) {
            palette.push(ch);
        }
    }
    palette.push(' ');
    println!("{}", palette.len());
    print!("███████████████████████████████████\n█");
    for y in 0..10 {
        for x in 0..33 {
            print!("{}", palette[y * 33 + x]);
        }
        print!("█\n█");
    }
    print!("██████████████████████████████████\n");
    return palette;
}

struct Matrix {
    width: usize,
    height: usize,
    contents: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(w: usize, h: usize) -> Matrix {
        let contents = vec![vec![0.0; h]; w];
        Matrix {
            width: w,
            height: h,
            contents,
        }
    }

    fn from_image_part(
        image: &image::RgbImage,
        x_pos: usize,
        y_pos: usize,
        width: usize,
        height: usize,
    ) -> Matrix {
        let mut img_mat = Matrix::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let p = image.get_pixel((x + x_pos * width) as u32, (y + y_pos * height) as u32);
                // print!("{} ", &Rgb(p.0[0], p.0[1], p.0[2]).bg_string());
                img_mat.set(x, y, p.0[0] as f64);
            }
            // print!("{}\r\n", termion::color::Reset.bg_str());
        }
        img_mat
    }

    fn from_1d(arr: &[f64], w: usize, h: usize) -> Matrix {
        let mut mat = Matrix::new(w, h);
        for y in 0..h {
            for x in 0..w {
                mat.set(x, y, arr[y * w + x]);
            }
        }
        mat
    }

    fn set(&mut self, x: usize, y: usize, val: f64) {
        self.contents[y][x] = val;
    }
    fn get(&self, x: usize, y: usize) -> f64 {
        self.contents[y][x]
    }

    fn max(&self) -> f64 {
        let mut current_max = -10000000.0;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) > current_max {
                    current_max = self.get(x, y);
                }
            }
        }
        current_max
    }
    fn min(&self) -> f64 {
        let mut current_min = 10000000.0;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) < current_min {
                    current_min = self.get(x, y);
                }
            }
        }
        current_min
    }

    fn abs_difference(&self, other: &Matrix, weights: &Matrix) -> f64 {
        let mut total_diff = 0.0;
        for x in 0..self.width {
            for y in 0..self.height {
                let diff = (self.get(x, y) - other.get(x, y)).abs();
                total_diff += diff / weights.get(x, y);
            }
        }
        // println!(
        //     "{}\n{}\n{}\n------------------------------------------",
        //     self, other, total_diff
        // );
        return total_diff;
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut buf: String = "".to_string();
        let max_val = self.max() + 0.01;
        let min_val = self.min();
        for y in 0..self.height {
            for x in 0..self.width {
                let val = self.get(x, y);
                let val_norm = (val - min_val) / (max_val - min_val);
                let val8 = (val_norm * 255.0) as u8;
                let antival = ((1.0 - val_norm) * 255.0) as u8;
                buf.push_str(&format!(
                    "{}{}{:05.02} ",
                    Rgb(val8, val8, val8).bg_string(),
                    Rgb(antival, antival, antival).fg_string(),
                    val,
                ));
            }
            buf.push_str(&format!(
                "{}{}\n",
                termion::color::Reset.fg_str(),
                termion::color::Reset.bg_str()
            ));
        }
        write!(f, "{}", buf)
    }
}

// fuck you, rust
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

// fuck you again
fn cos(x: f64) -> f64 {
    x.cos()
}

fn forward_dct_single_point(input: &Matrix, u: usize, v: usize) -> f64 {
    let cu = if u == 0 {
        1.0 / sqrt(input.width as f64)
        // 1.0 / sqrt(2.0)
    } else {
        sqrt(2.0) / sqrt(input.width as f64)
        // 1.0
    };
    let cv = if v == 0 {
        1.0 / sqrt(input.height as f64)
        // 1.0 / sqrt(2.0)
    } else {
        sqrt(2.0) / sqrt(input.height as f64)
        // 1.0
    };
    let mut result = 0.0;
    for x in 0..input.width {
        for y in 0..input.height {
            result += input.get(x, y)
                * cos(((2.0 * x as f64 + 1.0) * u as f64 * PI) / (2.0 * input.width as f64))
                * cos(((2.0 * y as f64 + 1.0) * v as f64 * PI) / (2.0 * input.height as f64));
        }
    }
    result * cu * cv
}

// fn inverse_dct_single_point<const W: usize, const H: usize>(
//     input: &Matrix<W, H>,
//     x: usize,
//     y: usize,
// ) -> f64 {
//     let mut result = 0.0;
//     for u in 0..W {
//         for v in 0..H {
//             let cu = if u == 0 {
//                 1.0 / sqrt(W as f64)
//             } else {
//                 sqrt(2.0) / sqrt(W as f64)
//             };
//             let cv = if v == 0 {
//                 1.0 / sqrt(H as f64)
//             } else {
//                 sqrt(2.0) / sqrt(H as f64)
//             };

//             result += cu
//                 * cv
//                 * input.get(u, v)
//                 * cos(((2.0 * x as f64 + 1.0) * u as f64 * PI) / (2.0 * W as f64))
//                 * cos(((2.0 * y as f64 + 1.0) * v as f64 * PI) / (2.0 * H as f64));
//         }
//     }
//     return result;
// }

fn forward_dct(input: &Matrix) -> Matrix {
    let mut mat = Matrix::new(input.width, input.height);
    for x in 0..input.width {
        for y in 0..input.height {
            mat.set(x, y, forward_dct_single_point(input, x, y));
        }
    }
    mat
}

// fn inverse_dct<const W: usize, const H: usize>(input: &Matrix<W, H>) -> Matrix<W, H> {
//     let mut mat = Matrix::new();
//     for x in 0..W {
//         for y in 0..H {
//             mat.set(x, y, inverse_dct_single_point(input, x, y));
//         }
//     }
//     mat
// }

fn create_calibration_matrices(
    dct: bool,
    x_cells: usize,
    y_cells: usize,
) -> (Vec<Matrix>, usize, usize) {
    let img = ImageReader::open("calibration.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    // println!("{}", img_mat);
    // let dct_mat = forward_dct(&img_mat);
    // let inv_dct_mat = forward_dct(&dct_mat);
    // println!("{}", dct_mat);
    // println!("{}", inv_dct_mat);

    let mut matrices = Vec::new();

    let cell_w = img.width() as usize / x_cells;
    let cell_h = img.height() as usize / y_cells;

    for map_y in 0..y_cells {
        for map_x in 0..x_cells {
            let img_mat = Matrix::from_image_part(&img, map_x, map_y, cell_w, cell_h);
            let dct_mat = forward_dct(&img_mat);
            matrices.push(if dct { dct_mat } else { img_mat });
        }
    }
    (matrices, cell_w, cell_h)
}
fn get_image_dominant_colours(img: &image::RgbImage, x0: usize, y0: usize) -> (Colour, Colour) {
    let mut colour_count = HashMap::<Colour, usize>::new();
    for y in y0..(y0 + 16) {
        for x in x0..(x0 + 7) {
            let pixel = img.get_pixel(x as u32, y as u32);
            let colour = Colour::from_rgb(pixel.0[0], pixel.0[1], pixel.0[2]);
            if colour_count.contains_key(&colour) {
                *colour_count.get_mut(&colour).unwrap() += 1;
            } else {
                colour_count.insert(colour, 1);
            }
        }
    }

    (Colour::from_rgb(0, 0, 0), Colour::from_rgb(0, 0, 0))
}

fn textify_dct(
    img: &image::RgbImage,
    matrices: &Vec<Matrix>,
    palette: &Vec<char>,
    cell_w: usize,
    cell_h: usize,
) {
    let x_cells = img.width() as usize / cell_w;
    let y_cells = img.height() as usize / cell_h;
    #[rustfmt::skip]
    let weights_1d_arr = [
        3.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0,
        1.0, 1.0, 1.0, 4.0, 5.0, 6.0, 7.0,
        2.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0,
        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
    ];

    let weights = Matrix::from_1d(&weights_1d_arr, cell_w, cell_h);

    for chunk_y in 0..y_cells {
        for chunk_x in 0..x_cells {
            // let (colour_a, colour_b) = get_image_dominant_colours(&img, chunk_x * 7, chunk_y * 16);
            let img_mat = Matrix::from_image_part(&img, chunk_x, chunk_y, cell_w, cell_h);
            let dct_mat = forward_dct(&img_mat);
            let mut min_idx = 0;
            let mut min_err = 1000000000000000.0;
            for (i, mat) in matrices.iter().enumerate() {
                let err = dct_mat.abs_difference(mat, &weights);
                if err < min_err {
                    min_err = err;
                    min_idx = i;
                }
            }
            print!("{}", palette[min_idx]);
        }
        println!();
    }
}

fn textify_spatial(img: &image::RgbImage, matrices: &Vec<Matrix<7, 16>>, palette: &Vec<char>) {
    let x_res = img.width() as usize / 7;
    let y_res = img.height() as usize / 16;
    let identity_weights = [1.0; 7 * 16];

    let weights = Matrix::from_1d(&identity_weights);

    for chunk_y in 0..y_res {
        for chunk_x in 0..x_res {
            let img_mat = Matrix::from_image_part(&img, chunk_x, chunk_y);
            let mut min_idx = 0;
            let mut min_err = 1000000000000000.0;
            for (i, mat) in matrices.iter().enumerate() {
                let err = img_mat.abs_difference(mat, &weights);
                if err < min_err {
                    min_err = err;
                    min_idx = i;
                }
            }
            print!("{}", palette[min_idx]);
        }
        println!();
    }
}

fn main() {
    let (palette, x_cells, y_cells) = print_calibration_image();
    let (dct_matrices, cell_w, cell_h) = create_calibration_matrices(true, x_cells, y_cells);
    let matrices = create_calibration_matrices(false, x_cells, y_cells);
    println!("{}", matrices.len());
    let img = ImageReader::open("test_landscape.png")
        .unwrap()
        .decode()
        .unwrap()
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
    //  12]
    textify_dct(&img, &dct_matrices, &palette, cell_w, cell_h);
    textify_spatial(&img, &matrices, &palette, cell_w, cell_h);
}
