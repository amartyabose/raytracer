use nalgebra as na;
use nalgebra::geometry as ng;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub orig: na::Point3<f32>,
    pub direction: na::Vector3<f32>
}

impl Ray {
    pub fn new(origin: na::Point3<f32>, dir: na::Vector3<f32>) -> Ray {
        let unit_dir = dir.normalize();
        Ray { orig: origin, direction: unit_dir }
    }

    pub fn at(&self, length: f32) -> na::Point3<f32> {
        ng::Translation3::from(length * self.direction) * self.orig
    }
}
