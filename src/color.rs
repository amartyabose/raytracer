use std::fmt;
use std::ops;

use nalgebra as na;
use serde::{Deserialize, Serialize};
use serde_derive::*;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Color {
    red: f32,
    green: f32,
    blue: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Color {
        Color {
            red: r,
            green: g,
            blue: b,
        }
    }

    pub fn from(v: na::Vector3<f32>) -> Color {
        Color {
            red: v[0],
            green: v[1],
            blue: v[2],
        }
    }

    pub fn clamp(&mut self) {
        if self.red > 1.0 {
            self.red = 1.0
        };
        if self.green > 1.0 {
            self.green = 1.0
        };
        if self.blue > 1.0 {
            self.blue = 1.0
        };
    }

    pub fn gamma_correction(&mut self) {
        self.red = self.red.sqrt();
        self.green = self.green.sqrt();
        self.blue = self.blue.sqrt();
    }
}

impl ops::Add for Color {
    type Output = Self;
    fn add(self, _rhs: Color) -> Color {
        Color {
            red: self.red + _rhs.red,
            green: self.green + _rhs.green,
            blue: self.blue + _rhs.blue,
        }
    }
}

impl ops::AddAssign for Color {
    fn add_assign(&mut self, _rhs: Color) {
        self.red += _rhs.red;
        self.green += _rhs.green;
        self.blue += _rhs.blue;
    }
}

impl ops::Div<f32> for Color {
    type Output = Self;
    fn div(self, _num: f32) -> Color {
        Color {
            red: self.red / _num,
            green: self.green / _num,
            blue: self.blue / _num,
        }
    }
}

impl ops::Mul<f32> for Color {
    type Output = Self;
    fn mul(self, _num: f32) -> Color {
        Color {
            red: self.red * _num,
            green: self.green * _num,
            blue: self.blue * _num,
        }
    }
}

impl ops::Mul<Color> for Color {
    type Output = Self;
    fn mul(self, _col: Color) -> Color {
        Color {
            red: self.red * _col.red,
            green: self.green * _col.green,
            blue: self.blue * _col.blue,
        }
    }
}

impl ops::MulAssign<Color> for Color {
    fn mul_assign(&mut self, _col: Color) {
        self.red *= _col.red;
        self.green *= _col.green;
        self.blue *= _col.blue;
    }
}

impl ops::Mul<Color> for f32 {
    type Output = Color;
    fn mul(self, _col: Color) -> Color {
        Color {
            red: _col.red * self,
            green: _col.green * self,
            blue: _col.blue * self,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let r = (u8::MAX as f32 * self.red) as u8;
        let g = (u8::MAX as f32 * self.green) as u8;
        let b = (u8::MAX as f32 * self.blue) as u8;
        write!(f, "{} {} {}", r, g, b)
    }
}
