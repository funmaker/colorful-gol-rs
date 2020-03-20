use std::sync::Arc;

use err_derive::Error;
use vulkano::{app_info_from_cargo_toml, OomError};
use vulkano::buffer::{BufferUsage, DeviceLocalBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetError, PersistentDescriptorSetBuildError};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Features, Queue, DeviceCreationError};
use vulkano::format::Format;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, LayersListError, InstanceCreationError};
use vulkano::pipeline::{ComputePipeline, ComputePipelineCreationError};
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, Surface, CapabilitiesError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::image::SwapchainImage;
use vulkano::memory::DeviceMemoryAllocError;
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use winit::window::Window;

use crate::shaders;


pub struct Renderer {
	pub instance: Arc<Instance>,
	pub surface: Option<Arc<Surface<Window>>>,
	
	device: Arc<Device>,
	queue: Arc<Queue>,
	pipeline: Arc<ComputePipeline<PipelineLayout<shaders::main::Layout>>>,
	swapchain: Option<(Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>)>,
	buffers: Option<(Arc<DeviceLocalBuffer<[f32]>>, Arc<DeviceLocalBuffer<[f32]>>)>,
	previous_frame_end: Option<Box<dyn GpuFuture>>,
	needs_regen: bool,
	needs_recreate_swapchain: bool,
	frame: u32,
}

impl Renderer {
	pub fn new(device: Option<usize>, debug: bool) -> Result<Renderer, RendererCreationError> {
		if debug {
			println!("List of Vulkan debugging layers available to use:");
			let mut layers = vulkano::instance::layers_list()?;
			while let Some(l) = layers.next() {
				println!("\t{}", l.name());
			}
		}
		
		let instance = {
			let app_infos = app_info_from_cargo_toml!();
			let extensions = InstanceExtensions { ext_debug_utils: true,
			                                      ..vulkano_win::required_extensions() };
			let layers = if debug {
				             vec!["VK_LAYER_LUNARG_standard_validation"]
			             } else {
				             vec![]
			             };
			Instance::new(Some(&app_infos), &extensions, layers)?
		};
		
		if debug {
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
				                                         
				                                         println!("{} {} {}: {}",
				                                                  msg.layer_prefix,
				                                                  ty,
				                                                  severity,
				                                                  msg.description);
			                                         });
		}
		
		println!("Devices:");
		for device in PhysicalDevice::enumerate(&instance) {
			println!("\t{}: {} api: {} driver: {}",
			         device.index(),
			         device.name(),
			         device.api_version(),
			         device.driver_version());
		}
		
		let physical = PhysicalDevice::enumerate(&instance)
		                              .skip(device.unwrap_or(0))
		                              .next()
		                              .ok_or(RendererCreationError::NoDevices)?;
		
		println!("\nUsing {}: {} api: {} driver: {}",
		         physical.index(),
		         physical.name(),
		         physical.api_version(),
		         physical.driver_version());
		
		if debug {
			for family in physical.queue_families() {
				println!("Found a queue family with {:?} queue(s)", family.queues_count());
			}
		}
		
		let queue_family = physical.queue_families()
		                           .find(|&q| q.supports_compute())
		                           .ok_or(RendererCreationError::NoQueue)?;
		
		let (device, mut queues) = Device::new(physical,
		                                       &Features::none(),
		                                       &DeviceExtensions { khr_storage_buffer_storage_class: true,
		                                                           khr_swapchain: true,
		                                                           ..DeviceExtensions::none() },
		                                       [(queue_family, 0.5)].iter().cloned())?;
		
		let queue = queues.next().ok_or(RendererCreationError::NoQueue)?;
		
		let shader = shaders::main::Shader::load(device.clone())?;
		let pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())?);
		
		let previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
		
		Ok(Renderer {
			instance,
			device,
			queue,
			pipeline,
			previous_frame_end,
			swapchain: None,
			buffers: None,
			surface: None,
			needs_regen: true,
			needs_recreate_swapchain: false,
			frame: 0,
		})
	}
	
	pub fn create_swapchain(&mut self, surface: Arc<Surface<Window>>) -> Result<(), RendererSwapchainError> {
		let caps = surface.capabilities(self.device.physical_device())?;
		let dimensions = caps.current_extent.unwrap_or(caps.min_image_extent);
		let alpha = caps.supported_composite_alpha.iter().next().unwrap();
		let format = caps.supported_formats
		                 .iter()
		                 .find(|format| format.0 == Format::B8G8R8A8Unorm || format.0 == Format::R8G8B8A8Unorm)
		                 .expect("UNorm format not supported on the surface");
		
		self.swapchain = Some(Swapchain::new(self.device.clone(),
		                                     surface.clone(),
		                                     caps.min_image_count,
		                                     format.0,
		                                     dimensions,
		                                     1,
		                                     caps.supported_usage_flags,
		                                     &self.queue,
		                                     SurfaceTransform::Identity,
		                                     alpha,
		                                     PresentMode::Fifo,
		                                     FullscreenExclusive::Allowed,
		                                     false,
		                                     format.1)?);
		
		self.regenerate_buffers(dimensions)?;
		self.surface = Some(surface);
		
		Ok(())
	}
	
	fn regenerate_buffers(&mut self, dimensions: [u32; 2]) -> Result<(), DeviceMemoryAllocError> {
		self.buffers = Some((
			DeviceLocalBuffer::array(self.device.clone(),
			                         (dimensions[0] * dimensions[1]) as usize,
			                         BufferUsage::all(),
			                         Some(self.queue.family()))?,
			DeviceLocalBuffer::array(self.device.clone(),
			                         (dimensions[0] * dimensions[1]) as usize,
			                         BufferUsage::all(),
			                         Some(self.queue.family()))?,
		));
		
		self.needs_regen = true;
		
		Ok(())
	}
	
	pub fn recreate_swapchain(&mut self) {
		self.needs_recreate_swapchain = true;
	}
	
	pub fn regenerate(&mut self) {
		self.needs_regen = true;
	}
	
	pub fn render(&mut self) -> Result<(), RenderError> {
		self.previous_frame_end.as_mut().unwrap().cleanup_finished();
		
		if self.needs_recreate_swapchain {
			let dimensions = self.surface.as_ref().unwrap().window().inner_size().into();
			
			self.swapchain = Some(self.swapchain.as_ref().unwrap().0.recreate_with_dimensions(dimensions)
			                          .map_err(|err| match err {
				                          SwapchainCreationError::UnsupportedDimensions => RenderError::Retry,
				                          err => RenderError::from(err),
			                          })?);
			
			self.regenerate_buffers(dimensions)?;
			self.needs_recreate_swapchain = false;
		}
		
		let (ref mut swapchain, ref mut images) = self.swapchain.as_mut().ok_or(RenderError::NoSwapchain)?;
		let buffers = self.buffers.as_mut().ok_or(RenderError::NoSwapchain)?;
		let dimensions = swapchain.dimensions();
		
		let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
			                                              Ok(x) => Ok(x),
			                                              Err(AcquireError::OutOfDate) => {
				                                              self.needs_recreate_swapchain = true;
				                                              Err(RenderError::Retry)
			                                              },
			                                              Err(err) => Err(RenderError::from(err)),
		                                              }?;
		
		if suboptimal {
			eprintln!("Suboptimal");
			self.needs_recreate_swapchain = true;
		}
		
		if image_num > 2 {
			eprintln!("Acquire_next_image returned {}! Skipping render.", image_num);
			self.needs_recreate_swapchain = true;
			return Err(RenderError::Retry);
		}
		
		let layout = self.pipeline.layout()
		                          .descriptor_set_layout(0)
		                          .expect("No set 0 in shader.");
		
		std::mem::swap(&mut buffers.0, &mut buffers.1);
		
		let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
		                                           .add_buffer(buffers.0.clone())?
		                                           .add_buffer(buffers.1.clone())?
		                                           .add_image(images[image_num].clone())?
		                                           .build()?);
		
		let command_buffer = AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())
			.unwrap()
			.dispatch([dimensions[0] / 8 + 1, dimensions[1] / 8 + 1, 1],
			          self.pipeline.clone(),
			          set.clone(),
			          (dimensions[0], dimensions[1], self.frame, if self.needs_regen {1_u32} else {0_u32}))
			.unwrap()
			.build()
			.unwrap();
		
		let future = self.previous_frame_end.take()
		                                    .unwrap()
		                                    .join(acquire_future)
		                                    .then_execute(self.queue.clone(), command_buffer)
		                                    .unwrap()
		                                    .then_swapchain_present(self.queue.clone(), swapchain.clone(), image_num)
		                                    .then_signal_fence_and_flush();
		
		self.frame += 1;
		self.needs_regen = false;
		
		match future {
			Ok(future) => {
				self.previous_frame_end = Some(Box::new(future) as Box<_>);
				Ok(())
			},
			Err(FlushError::OutOfDate) => {
				self.recreate_swapchain();
				self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
				Ok(())
			},
			Err(err) => {
				self.previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
				Err(RenderError::from(err))
			},
		}
	}
}


#[derive(Debug, Error)]
pub enum RendererCreationError {
	#[error(display = "No devices available.")]
	NoDevices,
	#[error(display = "No compute queue available.")]
	NoQueue,
	#[error(display = "{}", _0)]
	LayersListError(#[error(source)] LayersListError),
	#[error(display = "{}", _0)]
	InstanceCreationError(#[error(source)] InstanceCreationError),
	#[error(display = "{}", _0)]
	DeviceCreationError(#[error(source)] DeviceCreationError),
	#[error(display = "{}", _0)]
	OomError(#[error(source)] OomError),
	#[error(display = "{}", _0)]
	ComputePipelineCreationError(#[error(source)] ComputePipelineCreationError),
}

#[derive(Debug, Error)]
pub enum RendererSwapchainError {
	#[error(display = "{}", _0)]
	CapabilitiesError(#[error(source)] CapabilitiesError),
	#[error(display = "{}", _0)]
	SwapchainCreationError(#[error(source)] SwapchainCreationError),
	#[error(display = "{}", _0)]
	DeviceMemoryAllocError(#[error(source)] DeviceMemoryAllocError),
}

#[derive(Debug, Error)]
pub enum RenderError {
	#[error(display = "Try again.")]
	Retry,
	#[error(display = "Swapchain not initialized.")]
	NoSwapchain,
	#[error(display = "{}", _0)]
	CapabilitiesError(#[error(source)] CapabilitiesError),
	#[error(display = "{}", _0)]
	AcquireError(#[error(source)] AcquireError),
	#[error(display = "{}", _0)]
	SwapchainCreationError(#[error(source)] SwapchainCreationError),
	#[error(display = "{}", _0)]
	PersistentDescriptorSetError(#[error(source)] PersistentDescriptorSetError),
	#[error(display = "{}", _0)]
	PersistentDescriptorSetBuildError(#[error(source)] PersistentDescriptorSetBuildError),
	#[error(display = "{}", _0)]
	FlushError(#[error(source)] FlushError),
	#[error(display = "{}", _0)]
	DeviceMemoryAllocError(#[error(source)] DeviceMemoryAllocError),
}
