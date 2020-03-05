use std::error::Error;
use std::sync::Arc;

use vulkano::app_info_from_cargo_toml;
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::ComputePipeline;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder, Fullscreen};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

#define M_PI 3.1415926535897932384626433832795

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InCells {
    float data[];
} in_buf;

layout(set = 0, binding = 1) buffer OutCells {
    float data[];
} out_buf;

layout( push_constant ) uniform PConst {
  uint width;
  uint height;
  uint frame;
  bool reset;
} pconst;

layout(set = 0, binding = 2, rgba8) uniform writeonly image2D out_image;

//from https://www.shadertoy.com/view/XsGXDd
#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
#define HASHSCALE4 vec4(1031, .1030, .0973, .1099)

float hash13(vec3 p3) {
    p3  = fract(p3 * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float getAtIdx(vec3 v, int i) {
     if (i == 0) return v.x;
    else if (i == 1) return v.y;
    else if (i == 2) return v.z;
    else return 0.;
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    uint index = (gl_GlobalInvocationID.x % pconst.width)
               + (gl_GlobalInvocationID.y % pconst.height) * pconst.width;

    if(pconst.reset) out_buf.data[index] = hash13(vec3(pos, pconst.frame)) * 2;
    else {
        uint neighs = 0;
        vec3 hues = vec3(0);
        for(int y = pos.y - 1; y <= pos.y + 1; y++) {
            for(int x = pos.x - 1; x <= pos.x + 1; x++) {
                uint ngIndex = (x % pconst.width) + (y % pconst.height) * pconst.width;
                if(in_buf.data[ngIndex] >= 1) {
                    neighs++;
                    hues.yz = hues.xy;
                    hues.x = in_buf.data[ngIndex];
                }
            }
        }

        bool alive = in_buf.data[index] >= 1;

        if(!alive && neighs == 3) out_buf.data[index] = getAtIdx(hues, int(floor(mod(hash13(vec3(pos, pconst.frame)), 3. ))));
        else if(alive && (neighs < 3 || neighs > 4)) out_buf.data[index] = in_buf.data[index] - 1;
        else out_buf.data[index] = in_buf.data[index];
    }

    vec4 color = vec4(hsv2rgb(vec3(
        fract(out_buf.data[index]),
        1.0,
        out_buf.data[index] > 1 ? 1 : 0.25
    )), 1.0);
    imageStore(out_image, pos, color);
}"
    }
}

fn generate_buffers<'a, I>(device: Arc<Device>, dimensions: [u32; 2], queue_families: I)
                           -> Result<(Arc<DeviceLocalBuffer<[f32]>>, Arc<DeviceLocalBuffer<[f32]>>), Box<dyn Error>>
                           where I: IntoIterator<Item = QueueFamily<'a>> + Clone {
    Ok((
        DeviceLocalBuffer::array(device.clone(),
                                 (dimensions[0] * dimensions[1]) as usize,
                                 BufferUsage::all(),
                                 queue_families.clone())?,
        DeviceLocalBuffer::array(device.clone(),
                                 (dimensions[0] * dimensions[1]) as usize,
                                 BufferUsage::all(),
                                 queue_families.clone())?,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("List of Vulkan debugging layers available to use:");
    let mut layers = vulkano::instance::layers_list().unwrap();
    while let Some(l) = layers.next() {
        println!("\t{}", l.name());
    }

    let instance = {
        let app_infos = app_info_from_cargo_toml!();
        let extensions = InstanceExtensions { ext_debug_utils: true,
                                              ..vulkano_win::required_extensions() };
        let layers = vec!["VK_LAYER_LUNARG_standard_validation"];
        Instance::new(Some(&app_infos), &extensions, layers)?
    };

    let severity = MessageSeverity { error:       true,
                                     warning:     true,
                                     information: false,
                                     verbose:     true, };

    let ty = MessageType::all();

    let _debug_callback = DebugCallback::new(&instance, severity, ty, |msg| {
                              let severity = if msg.severity.error {
                                  "error"
                              } else if msg.severity.warning {
                                  "warning"
                              } else if msg.severity.information {
                                  "information"
                              } else if msg.severity.verbose {
                                  "verbose"
                              } else {
                                  panic!("no-impl");
                              };

                              let ty = if msg.ty.general {
                                  "general"
                              } else if msg.ty.validation {
                                  "validation"
                              } else if msg.ty.performance {
                                  "performance"
                              } else {
                                  panic!("no-impl");
                              };

                              println!("{} {} {}: {}", msg.layer_prefix, ty, severity, msg.description);
                          }).ok();

    println!("Devices:");
    for device in PhysicalDevice::enumerate(&instance) {
        println!("\t{}: {} api: {} driver: {}",
                 device.index(),
                 device.name(),
                 device.api_version(),
                 device.driver_version());
    }

    let physical = PhysicalDevice::enumerate(&instance).next().expect("No devices available");
    println!("Using {}", physical.index());

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().with_transparent(true)
                                      .with_title("Colorful Game Of Life")
                                      .build_vk_surface(&event_loop, instance.clone())?;
    let window = surface.window();

    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    let queue_family = physical.queue_families()
                               .find(|&q| q.supports_compute())
                               .expect("No device compute queue family available");

    let (device, mut queues) = Device::new(physical,
                                           &Features::none(),
                                           &DeviceExtensions { khr_storage_buffer_storage_class: true,
                                                               khr_swapchain: true,
                                                               ..DeviceExtensions::none() },
                                           [(queue_family, 0.5)].iter().cloned())?;

    let queue = queues.next().expect("No queues available");
    let mut dimensions;

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical)?;
        dimensions = caps.current_extent.unwrap_or(window.inner_size().into());
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats
                         .iter()
                         .find(|format| format.0 == Format::B8G8R8A8Unorm || format.0 == Format::R8G8B8A8Unorm)
                         .expect("UNorm format not supported on the surface");

        Swapchain::new(device.clone(),
                       surface.clone(),
                       caps.min_image_count,
                       format.0,
                       dimensions,
                       1,
                       caps.supported_usage_flags,
                       &queue,
                       SurfaceTransform::Identity,
                       alpha,
                       PresentMode::Fifo,
                       FullscreenExclusive::Allowed,
                       false,
                       format.1)?
    };

    #[rustfmt::skip]
    let mut cells = generate_buffers(device.clone(), dimensions.clone(), Some(queue_family))?;

    let shader = cs::Shader::load(device.clone())?;
    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())?);

    let mut recreate_swapchain = false;
    let mut regenerate = true;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);
    let mut frame = 0;

    event_loop.run(move |event, _, control_flow| {
                  *control_flow = ControlFlow::Poll;

                  match event {
                      Event::WindowEvent { event: WindowEvent::CloseRequested,
                                           .. } => {
                          *control_flow = ControlFlow::Exit;
                      },

                      Event::WindowEvent {
                          event: WindowEvent::KeyboardInput {
                              input: KeyboardInput {
                                  virtual_keycode: Some(VirtualKeyCode::R),
                                  state: ElementState::Pressed,
                                  ..
                              },
                              ..
                          },
                          ..
                      } => {
                          println!("Regen request");
                          regenerate = true;
                      },

                      Event::WindowEvent {
                          event: WindowEvent::KeyboardInput {
                              input: KeyboardInput {
                                  virtual_keycode: Some(VirtualKeyCode::Q),
                                  state: ElementState::Pressed,
                                  ..
                              },
                              ..
                          },
                          ..
                      } => {
                          *control_flow = ControlFlow::Exit;
                      },

                      Event::WindowEvent {
                          event: WindowEvent::KeyboardInput {
                              input: KeyboardInput {
                                  virtual_keycode: Some(VirtualKeyCode::F),
                                  state: ElementState::Pressed,
                                  ..
                              },
                              ..
                          },
                          ..
                      } => {
                          let window = surface.window();

                          if let None = window.fullscreen() {
                              window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
                              window.set_cursor_visible(false);
                          } else {
                              window.set_fullscreen(None);
                              window.set_cursor_visible(true);
                          }
                      },

                      Event::WindowEvent { event: WindowEvent::Resized(_),
                                           .. } => {
                          recreate_swapchain = true;
                      },

                      Event::RedrawRequested(_) | Event::RedrawEventsCleared => {
                          loop {
                              previous_frame_end.as_mut().unwrap().cleanup_finished();

                              if recreate_swapchain {
                                  dimensions = surface.window().inner_size().into();

                                  let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                                      Ok(r) => r,
                                      // This error tends to happen when the user is manually resizing the window.
                                      // Simply restarting the loop is the easiest way to fix this issue.
                                      Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                                      Err(err) => {
                                          eprintln!("Failed to recreate swapchain: {:?}", err);
                                          *control_flow = ControlFlow::Exit;
                                          return;
                                      },
                                  };

                                  cells = generate_buffers(device.clone(),
                                                           dimensions.clone(),
                                                           cells.0.queue_families().clone()).unwrap();

                                  swapchain = new_swapchain;
                                  images = new_images;
                                  regenerate = true;
                                  recreate_swapchain = false;
                              }

                              let (image_num, suboptimal, acquire_future) =
                                  match swapchain::acquire_next_image(swapchain.clone(), None) {
                                      Ok(r) => r,
                                      Err(AcquireError::OutOfDate) => {
                                          recreate_swapchain = true;
                                          continue;
                                      },
                                      Err(err) => panic!("Failed to acquire next image: {:?}", err),
                                  };

                              if image_num > 2 {
                                  eprintln!("Acquire_next_image returned {}! Skipping render.", image_num);
                                  return;
                              }

                              if suboptimal {
                                  recreate_swapchain = true;
                              }

                              let layout = compute_pipeline.layout()
                                                           .descriptor_set_layout(0)
                                                           .expect("No set 0 in shader.");

                              std::mem::swap(&mut cells.0, &mut cells.1);

                              #[rustfmt::skip]
                              let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                                  .add_buffer(cells.0.clone()).unwrap()
                                  .add_buffer(cells.1.clone()).unwrap()
                                  .add_image(images[image_num].clone()).unwrap()
                                  .build().unwrap());

                              #[rustfmt::skip]
                              let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
                                  .unwrap()
                                  .dispatch([dimensions[0] / 8 + 1, dimensions[1] / 8 + 1, 1],
                                            compute_pipeline.clone(),
                                            set.clone(),
                                            (dimensions[0], dimensions[1], frame, if regenerate {1_u32} else {0_u32}))
                                  .unwrap()
                                  .build()
                                  .unwrap();

                              let future = previous_frame_end.take()
                                                             .unwrap()
                                                             .join(acquire_future)
                                                             .then_execute(queue.clone(), command_buffer)
                                                             .unwrap()
                                                             .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                                                             .then_signal_fence_and_flush();

                              frame += 1;
                              regenerate = false;

                              match future {
                                  Ok(future) => {
                                      previous_frame_end = Some(Box::new(future) as Box<_>);
                                      // println!("Rendered! {}", image_num);
                                  },
                                  Err(FlushError::OutOfDate) => {
                                      recreate_swapchain = true;
                                      previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                                  },
                                  Err(e) => {
                                      println!("{:?}", e);
                                      previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                                  },
                              }

                              break;
                          }
                      },
                      _ => {},
                  }
              })
}
