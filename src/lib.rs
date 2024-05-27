use fmtstring::{Colour, FmtChar, FmtString, Ground};
use image::{imageops::FilterType, io::Reader as ImageReader};
use std::f64::consts::PI;
use termion::color::Rgb;

pub fn get_palette() -> Vec<char> {
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
    palette.extend_from_slice(&"⎻⎽⎺⎼".chars().collect::<Vec<_>>());
    palette
}

pub fn print_calibration_image() {
    let palette = get_palette();
    println!("{}", palette.len());
    print!("{}\n█", "█".repeat(37 * 2 + 2));
    for y in 0..9 {
        for x in 0..37 {
            use termion::color::*;
            print!("{}{}{} ", Fg(White), Bg(Black), palette[y * 37 + x]);
        }
        print!("█\n█");
        print!("{}█\n█", " ".repeat(37 * 2));
    }
    print!("{}\n", "█".repeat(37 * 2 + 1));
}

pub struct Matrix<const W: usize, const H: usize>([[f64; W]; H]);

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub fn new() -> Matrix<W, H> {
        Matrix { 0: [[0.0; W]; H] }
    }

    pub fn from_image_part(image: &image::RgbImage, x_pos: usize, y_pos: usize) -> Matrix<W, H> {
        let mut img_mat = Matrix::<W, H>::new();
        for y in 0..H {
            for x in 0..W {
                let p = image.get_pixel((x + x_pos * W) as u32, (y + y_pos * H) as u32);
                // print!("{} ", &Rgb(p.0[0], p.0[1], p.0[2]).bg_string());
                img_mat.set(x, y, p.0[0] as f64);
            }
            // print!("{}\r\n", termion::color::Reset.bg_str());
        }
        img_mat
    }

    pub fn from_image_part_with_colours(
        image: &image::RgbImage,
        x_pos: usize,
        y_pos: usize,
        col1: &Colour,
        col2: &Colour,
    ) -> Matrix<W, H> {
        let mut img_mat = Matrix::<W, H>::new();
        // let r = termion::color::Reset.bg_str();
        // println!(
        //     "{} {}{} {}",
        //     col1.to_string(fmtstring::Ground::Background),
        //     r,
        //     col2.to_string(fmtstring::Ground::Background),
        //     r
        // );
        for y in 0..H {
            for x in 0..W {
                let p = image.get_pixel((x + x_pos * W) as u32, (y + y_pos * H) as u32);
                // print!("{} ", &Rgb(p.0[0], p.0[1], p.0[2]).bg_string());
                let pixel_colour = Colour::from_rgb(p.0[0], p.0[1], p.0[2]);
                let col1_err = colour_diff(col1, &pixel_colour);
                let col2_err = colour_diff(col2, &pixel_colour);
                img_mat.set(x, y, if col1_err > col2_err { 255.0 } else { 0.0 });
            }
            // print!("{}\r\n", termion::color::Reset.bg_str());
        }
        // println!("\n{}", img_mat);
        // println!("\n\n");
        img_mat
    }

    pub fn from_1d(arr: [f64; 16 * 7]) -> Matrix<W, H> {
        let mut mat = Matrix::new();
        for y in 0..H {
            for x in 0..W {
                mat.set(x, y, arr[y * W + x]);
            }
        }
        mat
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.0[y][x] = val;
    }
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.0[y][x]
    }

    pub fn max(&self) -> f64 {
        let mut current_max = -10000000.0;
        for y in 0..H {
            for x in 0..W {
                if self.get(x, y) > current_max {
                    current_max = self.get(x, y);
                }
            }
        }
        current_max
    }
    fn min(&self) -> f64 {
        let mut current_min = 10000000.0;
        for y in 0..H {
            for x in 0..W {
                if self.get(x, y) < current_min {
                    current_min = self.get(x, y);
                }
            }
        }
        current_min
    }

    pub fn abs_difference(&self, other: &Matrix<W, H>, weights: &Matrix<W, H>) -> f64 {
        let mut total_diff = 0.0;
        for x in 0..W {
            for y in 0..H {
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

impl<const W: usize, const H: usize> std::fmt::Display for Matrix<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut buf: String = "".to_string();
        let max_val = self.max() + 0.01;
        let min_val = self.min();
        for y in 0..H {
            for x in 0..W {
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

fn forward_dct_single_point<const W: usize, const H: usize>(
    input: &Matrix<W, H>,
    u: usize,
    v: usize,
) -> f64 {
    let cu = if u == 0 {
        1.0 / sqrt(W as f64)
        // 1.0 / sqrt(2.0)
    } else {
        sqrt(2.0) / sqrt(W as f64)
        // 1.0
    };
    let cv = if v == 0 {
        1.0 / sqrt(H as f64)
        // 1.0 / sqrt(2.0)
    } else {
        sqrt(2.0) / sqrt(H as f64)
        // 1.0
    };
    let mut result = 0.0;
    for x in 0..W {
        for y in 0..H {
            result += input.get(x, y)
                * cos(((2.0 * x as f64 + 1.0) * u as f64 * PI) / (2.0 * W as f64))
                * cos(((2.0 * y as f64 + 1.0) * v as f64 * PI) / (2.0 * H as f64));
        }
    }
    result * cu * cv
}

fn inverse_dct_single_point<const W: usize, const H: usize>(
    input: &Matrix<W, H>,
    x: usize,
    y: usize,
) -> f64 {
    let mut result = 0.0;
    for u in 0..W {
        for v in 0..H {
            let cu = if u == 0 {
                1.0 / sqrt(W as f64)
            } else {
                sqrt(2.0) / sqrt(W as f64)
            };
            let cv = if v == 0 {
                1.0 / sqrt(H as f64)
            } else {
                sqrt(2.0) / sqrt(H as f64)
            };

            result += cu
                * cv
                * input.get(u, v)
                * cos(((2.0 * x as f64 + 1.0) * u as f64 * PI) / (2.0 * W as f64))
                * cos(((2.0 * y as f64 + 1.0) * v as f64 * PI) / (2.0 * H as f64));
        }
    }
    return result;
}

fn forward_dct<const W: usize, const H: usize>(input: &Matrix<W, H>) -> Matrix<W, H> {
    let mut mat = Matrix::new();
    for x in 0..W {
        for y in 0..H {
            mat.set(x, y, forward_dct_single_point(input, x, y));
        }
    }
    mat
}

fn inverse_dct<const W: usize, const H: usize>(input: &Matrix<W, H>) -> Matrix<W, H> {
    let mut mat = Matrix::new();
    for x in 0..W {
        for y in 0..H {
            mat.set(x, y, inverse_dct_single_point(input, x, y));
        }
    }
    mat
}

pub fn create_calibration_matrices<const W: usize, const H: usize>(dct: bool) -> Vec<Matrix<W, H>> {
    let img = ImageReader::open("/home/james/Build/dct-tiv/calibration.png")
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

    let x_res = img.width() as usize / W / 2;
    let y_res = img.height() as usize / H / 2;

    for map_y in 0..y_res {
        for map_x in 0..x_res {
            let img_mat = Matrix::from_image_part(&img, map_x * 2, map_y * 2);
            matrices.push(if dct { forward_dct(&img_mat) } else { img_mat });
        }
    }
    matrices
}

fn colour_diff(a: &Colour, b: &Colour) -> f32 {
    let Colour::Rgb {
        r: ar,
        g: ag,
        b: ab,
    } = a
    else {
        unreachable!()
    };
    let Colour::Rgb {
        r: br,
        g: bg,
        b: bb,
    } = b
    else {
        unreachable!()
    };
    let oklab_a = oklab::srgb_to_oklab(oklab::RGB::<u8>::new(*ar, *ag, *ab));
    let oklab_b = oklab::srgb_to_oklab(oklab::RGB::<u8>::new(*br, *bg, *bb));
    let dl = oklab_a.l - oklab_b.l;
    let da = oklab_a.a - oklab_b.a;
    let db = oklab_a.b - oklab_b.b;
    dl * dl + da * da + db * db
}

fn get_image_dominant_colours(img: &image::RgbImage, x0: usize, y0: usize) -> (Colour, Colour) {
    // let mut colour_count = HashMap::<Colour, usize>::new();
    // for y in y0..(y0 + 16) {
    //     for x in x0..(x0 + 7) {
    //         let pixel = img.get_pixel(x as u32, y as u32);
    //         let colour = Colour::from_rgb(pixel.0[0], pixel.0[1], pixel.0[2]);
    //         if colour_count.contains_key(&colour) {
    //             *colour_count.get_mut(&colour).unwrap() += 1;
    //         } else {
    //             colour_count.insert(colour, 1);
    //         }
    //     }
    // }

    let mut bit_we_care_about = [Colour::from_rgb(0, 0, 0); 7 * 16];
    for y in y0..(y0 + 16) {
        for x in x0..(x0 + 7) {
            let pixel = img.get_pixel(x as u32, y as u32);
            let colour = Colour::from_rgb(pixel.0[0], pixel.0[1], pixel.0[2]);
            bit_we_care_about[(y - y0) * 7 + (x - x0)] = colour;
        }
    }

    // let mut max_colours = (Colour::from_rgb(0, 0, 0), Colour::from_rgb(0, 0, 0));
    const SIZE: usize = 16;
    let mut space = [(oklab::srgb_to_oklab(oklab::RGB::new(0, 0, 0)), 0); SIZE * SIZE * SIZE];

    for idx in 0..(7 * 16) {
        let a = bit_we_care_about[idx];
        let Colour::Rgb { r, g, b } = a else {
            unreachable!()
        };
        let c = oklab::srgb_to_oklab(oklab::RGB::<u8>::new(r, g, b));
        let x = (c.l * SIZE as f32) as usize;
        let y = (c.a * SIZE as f32) as usize;
        let z = (c.b * SIZE as f32) as usize;

        space[z * SIZE * SIZE + y * SIZE + x].0.l += c.l;
        space[z * SIZE * SIZE + y * SIZE + x].0.a += c.a;
        space[z * SIZE * SIZE + y * SIZE + x].0.b += c.b;
        space[z * SIZE * SIZE + y * SIZE + x].1 += 1;
    }

    let m = space.iter().enumerate().max_by_key(|x| x.1 .1).unwrap();
    let m2 = space
        .iter()
        .enumerate()
        .filter(|x| x.0 != m.0)
        .max_by_key(|x| x.1 .1)
        .unwrap_or(m); // 2nd max

    let m = oklab::oklab_to_srgb(oklab::Oklab {
        l: m.1 .0.l / m.1 .1 as f32,
        a: m.1 .0.a / m.1 .1 as f32,
        b: m.1 .0.b / m.1 .1 as f32,
    });
    let m2 = oklab::oklab_to_srgb(oklab::Oklab {
        l: m2.1 .0.l / m2.1 .1 as f32,
        a: m2.1 .0.a / m2.1 .1 as f32,
        b: m2.1 .0.b / m2.1 .1 as f32,
    });
    let max_colours = (
        Colour::from_rgb(m.r, m.g, m.b),
        Colour::from_rgb(m2.r, m2.g, m2.b),
    );

    // let mut max_diff = -1.0;
    // for i in 0..(7 * 16) {
    //     for j in 0..(7 * 16) {
    //         if i == j {
    //             continue;
    //         }
    //         let col1 = bit_we_care_about[i];
    //         let col2 = bit_we_care_about[j];
    //         let diff = colour_diff(&col1, &col2);
    //         if diff > max_diff {
    //             max_diff = diff;
    //             max_colours = (col1, col2);
    //         }
    //     }
    // }
    let img_mat = Matrix::<7, 16>::from_image_part_with_colours(
        img,
        x0 / 7,
        y0 / 16,
        &max_colours.0,
        &max_colours.1,
    );

    // println!();
    // println!(
    //     "{}  {}  {}  {}",
    //     max_colours.0.to_string(Ground::Background),
    //     termion::color::Bg(termion::color::Reset),
    //     max_colours.1.to_string(Ground::Background),
    //     termion::color::Bg(termion::color::Reset)
    // );
    // for y in 0..16 {
    //     for x in 0..7 {
    //         print!(
    //             "{}  ",
    //             bit_we_care_about[y * 7 + x].to_string(Ground::Background)
    //         );
    //     }
    //     print!("{}    ", termion::color::Bg(termion::color::Reset));

    //     for x in 0..7 {
    //         let p = img_mat.get(x, y) as u8;
    //         let c = if p == 0 { max_colours.0 } else { max_colours.1 };
    //         print!("{}  ", c.to_string(Ground::Background));
    //     }
    //     print!("{}\n", termion::color::Bg(termion::color::Reset));
    // }

    // println!(
    //     "\n\n{:?}, {:?}\n{:?}\n\n",
    //     max_colours.0, max_colours.1, bit_we_care_about
    // );
    max_colours

    // (Colour::from_rgb(0, 0, 0), Colour::from_rgb(0, 0, 0))
}

pub fn textify_dct(
    img: &image::RgbImage,
    matrices: &Vec<Matrix<7, 16>>,
    palette: &Vec<char>,
) -> Vec<FmtString> {
    let x_res = img.width() as usize / 7;
    let y_res = img.height() as usize / 16;

    #[rustfmt::skip]
    let weights_1d_arr = [
        3.0,  1.0,  1.0,  3.0,  4.0,  5.0,  6.0,
        1.0,  1.0,  1.0,  4.0,  5.0,  6.0,  7.0,
        2.0,  2.0,  3.0,  5.0,  6.0,  7.0,  8.0,
        3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
        4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
        5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
        6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
        7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0,
        8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
        9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
    ];

    let weights = Matrix::from_1d(weights_1d_arr);

    let mut output = Vec::with_capacity(y_res);
    for chunk_y in 0..y_res {
        let mut line = FmtString::with_capacity(x_res);
        for chunk_x in 0..x_res {
            let (colour_a, colour_b) = get_image_dominant_colours(&img, chunk_x * 7, chunk_y * 16);
            let img_mat =
                Matrix::from_image_part_with_colours(&img, chunk_x, chunk_y, &colour_a, &colour_b);
            let dct_mat = forward_dct(&img_mat);
            // println!("{}", img_mat);
            let mut min_idx = 0;
            let mut min_err = 1000000000000000.0;
            for (i, mat) in matrices.iter().enumerate() {
                let err = dct_mat.abs_difference(mat, &weights);
                if err < min_err {
                    min_err = err;
                    min_idx = i;
                }
            }
            line.push(FmtChar {
                ch: palette[min_idx],
                fg: colour_b,
                bg: colour_a,
            });
            // println!("{}", palette[min_idx]);
            // print!(
            //     "{}{}{}",
            //     colour_b.to_string(fmtstring::Ground::Foreground),
            //     colour_a.to_string(fmtstring::Ground::Background),
            //     palette[min_idx]
            // );
        }
        output.push(line);
        // print!(
        //     "{}{}\n",
        //     termion::color::Reset.fg_str(),
        //     termion::color::Reset.bg_str()
        // );
    }
    output
}

pub fn textify_spatial(
    img: &image::RgbImage,
    matrices: &Vec<Matrix<7, 16>>,
    palette: &Vec<char>,
) -> Vec<FmtString> {
    let x_res = img.width() as usize / 7;
    let y_res = img.height() as usize / 16;
    let identity_weights = [1.0; 7 * 16];

    let weights = Matrix::from_1d(identity_weights);

    let mut output = Vec::with_capacity(y_res);
    for chunk_y in 0..y_res {
        let mut line = FmtString::with_capacity(x_res);
        for chunk_x in 0..x_res {
            let (colour_a, colour_b) = get_image_dominant_colours(&img, chunk_x * 7, chunk_y * 16);
            let img_mat =
                Matrix::from_image_part_with_colours(&img, chunk_x, chunk_y, &colour_a, &colour_b);
            let mut min_idx = 0;
            let mut min_err = 1000000000000000.0;
            for (i, mat) in matrices.iter().enumerate() {
                let err = img_mat.abs_difference(mat, &weights);
                if err < min_err {
                    min_err = err;
                    min_idx = i;
                }
            }
            line.push(FmtChar {
                ch: palette[min_idx],
                fg: colour_b,
                bg: colour_a,
            });
            // println!("{}", palette[min_idx]);
            // print!(
            //     "{}{}{}",
            //     colour_b.to_string(fmtstring::Ground::Foreground),
            //     colour_a.to_string(fmtstring::Ground::Background),
            //     palette[min_idx]
            // );
        }
        output.push(line);
        // print!(
        //     "{}{}\n",
        //     termion::color::Reset.fg_str(),
        //     termion::color::Reset.bg_str()
        // );
    }
    output
}
