use std::fs::File;
use std::io::Write;

use rand::Rng;

use itertools::Itertools;
use nalgebra as na;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_derive::*;
use serde_json::*;

mod color;
mod ray;

#[derive(Clone, Copy, Deserialize, Serialize)]
enum MaterialType {
    Lambertian,
    Metal(f32),
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct Material {
    material_type: MaterialType,
    color: color::Color,
}

fn random_unit_vector(rng: &mut rand::rngs::ThreadRng) -> na::Vector3<f32> {
    let x: f32 = rng.gen_range(-1f32, 1f32);
    let y: f32 = rng.gen_range(-1f32, 1f32);
    let z: f32 = rng.gen_range(-1f32, 1f32);

    na::Vector3::new(x, y, z).normalize()
}

fn scatter(
    rng: &mut rand::rngs::ThreadRng,
    in_ray: ray::Ray,
    intersection_pt: na::Point3<f32>,
    normal_vec: na::Vector3<f32>,
    material: Material,
) -> ray::Ray {
    match material.material_type {
        MaterialType::Lambertian => {
            ray::Ray::new(intersection_pt, random_unit_vector(rng) + normal_vec)
        }
        MaterialType::Metal(fuzziness) => ray::Ray::new(
            intersection_pt,
            in_ray.direction - 2f32 * normal_vec.dot(&in_ray.direction) * normal_vec
                + fuzziness * random_unit_vector(rng),
        ),
    }
}

trait Object {
    fn intersect(&self, ray: &ray::Ray) -> Option<f32>;
    fn normal(&self, pt: na::Point3<f32>) -> na::Vector3<f32>;
    fn get_color(&self) -> color::Color;
    fn get_material(&self) -> Material;
}

#[derive(Clone, Copy)]
struct Sphere {
    centre: na::Point3<f32>,
    radius: f32,
    material: Material,
}

impl Object for Sphere {
    fn intersect(&self, ray: &ray::Ray) -> Option<f32> {
        let oc = ray.orig - self.centre;
        let c = oc.norm().powi(2) - self.radius.powi(2);
        let half_b = oc.dot(&ray.direction);
        let determinant = half_b.powi(2) - c;
        if determinant < 0.0 {
            None
        } else {
            let val = -half_b - determinant.sqrt();
            if val >= 0.0 {
                Some(val)
            } else {
                None
            }
        }
    }

    fn normal(&self, pt: na::Point3<f32>) -> na::Vector3<f32> {
        (pt - self.centre).normalize()
    }

    fn get_color(&self) -> color::Color {
        self.material.color
    }

    fn get_material(&self) -> Material {
        self.material
    }
}

fn nearest_intersection<'a>(
    ray: &ray::Ray,
    objs: &'a [Box<dyn Object + Sync>],
) -> Option<(
    na::Point3<f32>,
    na::Vector3<f32>,
    &'a Box<dyn Object + Sync>,
)> {
    let mut nearest_obj: Option<&Box<dyn Object + Sync>> = None;
    let mut tmin: Option<f32> = None;
    for o in objs {
        if let Some(t) = o.intersect(ray) {
            if tmin.is_none() || t < tmin.unwrap() {
                tmin = Some(t);
                nearest_obj = Some(o);
            }
        }
    }

    match (nearest_obj, tmin) {
        (Some(o), Some(t)) => Some((ray.at(t), o.normal(ray.at(t)), o)),
        (_, _) => None,
    }
}

fn raytracing_ppm<F>(
    outputfile: &str,
    aspect_ratio: f32,
    img_height: u32,
    viewport_height: f32,
    ray_color: F,
) -> std::io::Result<()>
where
    F: Fn(ray::Ray, &mut rand::rngs::ThreadRng) -> color::Color + Sync,
{
    let img_width: u32 = (img_height as f32 * aspect_ratio) as u32;

    let viewport_width: f32 = viewport_height * aspect_ratio;
    let focal_length: f32 = 1.0;

    let origin: na::Vector3<f32> = na::Vector3::new(0.0, 0.0, 0.0);
    let vertical: na::Vector3<f32> = na::Vector3::y() * viewport_height as f32;
    let horizontal: na::Vector3<f32> = na::Vector3::x() * viewport_width as f32;
    let lower_left_corner = na::Vector3::new(0.0, 0.0, 0.0)
        - vertical / 2.0
        - horizontal / 2.0
        - na::Vector3::z() * focal_length;

    let samples_per_pixel = 500u32;

    let colors: Vec<color::Color> = (0..img_height)
        .rev()
        .cartesian_product(0..img_width)
        .collect::<Vec<(u32, u32)>>()
        .into_par_iter()
        .map(|x| -> color::Color {
            let mut col = color::Color::new(0.0, 0.0, 0.0);
            let mut rng = rand::thread_rng();
            for _ in 0..samples_per_pixel {
                let mut r: f32 = rng.gen();
                let u: f32 = (x.1 as f32 + r) / (img_width - 1) as f32;
                r = rng.gen();
                let v: f32 = (x.0 as f32 + r) / (img_height - 1) as f32;
                let current_ray = ray::Ray::new(
                    na::Point3::from(origin),
                    lower_left_corner + u * horizontal + v * vertical - origin,
                );
                col += ray_color(current_ray, &mut rng);
            }
            col / samples_per_pixel as f32
        })
        .collect();

    let mut outfile = File::create(outputfile)?;
    writeln!(outfile, "P3\n{} {}\n{}", img_width, img_height, u8::MAX)?;

    for mut color in colors {
        color.gamma_correction();
        color.clamp();
        writeln!(outfile, "{}", color)?;
    }

    Ok(())
}

fn raytracing(
    aspect_ratio: f32,
    height: u32,
    view_port_height: f32,
    objects: &[Box<dyn Object + Sync>],
    filename: &str,
) {
    match raytracing_ppm(
        filename,
        aspect_ratio,
        height,
        view_port_height,
        |r: ray::Ray, rng: &mut rand::rngs::ThreadRng| -> color::Color {
            let mut used_ray = r;
            let mut count: i32 = 0;
            let max_depth = 20;
            let mut col = color::Color::new(1f32, 1f32, 1f32);
            for _ in 0..max_depth {
                if let Some((intersect_pt, normal_vec, nearest_obj)) =
                    nearest_intersection(&used_ray, &objects)
                {
                    used_ray = scatter(
                        rng,
                        used_ray,
                        intersect_pt,
                        normal_vec,
                        nearest_obj.get_material(),
                    );
                    col *= nearest_obj.get_color();
                    count += 1;
                } else {
                    break;
                }
            }
            if count == max_depth + 1 {
                color::Color::new(0.0, 0.0, 0.0)
            } else {
                let t = 0.5 * (used_ray.direction[1] + 1.0);
                col * ((1.0f32 - t) * color::Color::new(1f32, 1f32, 1f32)
                    + t * color::Color::new(0.5f32, 0.7f32, 1f32))
            }
        },
    ) {
        Ok(()) => println!("Printed {}", filename),
        Err(e) => println!("Error happened while printing {}: {}", filename, e),
    }
}

fn main() -> std::io::Result<()> {
    let aspect_ratio: f32 = 16.0 / 9.0;
    let height: u32 = 256;

    let view_port_height: f32 = 2.0;

    let objects: Vec<Box<dyn Object + Sync>> = vec![
        Box::new(Sphere {
            centre: na::Point3::new(0.0, -100.5, -1.0),
            radius: 100f32,
            material: Material {
                material_type: MaterialType::Lambertian,
                color: color::Color::new(0.8f32, 0.8f32, 0.0),
            },
        }),
        Box::new(Sphere {
            centre: na::Point3::new(0.0, 0.0, -1.0),
            radius: 0.5f32,
            material: Material {
                material_type: MaterialType::Lambertian,
                color: color::Color::new(0.7f32, 0.3f32, 0.3f32),
            },
        }),
        Box::new(Sphere {
            centre: na::Point3::new(-1.0, 0.0, -1.0),
            radius: 0.5f32,
            material: Material {
                material_type: MaterialType::Metal(0.15),
                color: color::Color::new(0.8f32, 0.8f32, 0.8f32),
            },
        }),
        Box::new(Sphere {
            centre: na::Point3::new(1.0, 0.0, -1.0),
            radius: 0.5f32,
            material: Material {
                material_type: MaterialType::Metal(0.0),
                color: color::Color::new(0.8f32, 0.6f32, 0.2f32),
            },
        }),
    ];

    raytracing(
        aspect_ratio,
        height,
        view_port_height,
        &objects,
        "05_spheres_pic.ppm",
    );

    let mat1 = MaterialType::Lambertian;
    let mat2 = MaterialType::Metal(0.5);
    println!("mat1 = {}", to_string(&mat1)?);
    println!("mat2 = {}", to_string(&mat2)?);

    Ok(())
}
