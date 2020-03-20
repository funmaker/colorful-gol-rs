use err_derive::Error;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder, Fullscreen};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use vulkano_win::{VkSurfaceBuild, CreationError};

use crate::renderer::{Renderer, RendererSwapchainError, RenderError};

pub struct Application {
	renderer: Renderer,
	event_loop: EventLoop<()>,
}

#[derive(Debug, Error)]
pub enum ApplicationCreationError {
	#[error(display = "{}", _0)]
	RendererSwapchainError(#[error(source)] RendererSwapchainError),
	#[error(display = "{}", _0)]
	WindowBuilderErrro(#[error(source)] CreationError),
}

impl Application {
	pub fn new(mut renderer: Renderer) -> Result<Application, ApplicationCreationError> {
		let event_loop = EventLoop::new();
		
		let surface = WindowBuilder::new().with_transparent(true)
		                                  .with_inner_size(PhysicalSize::new(1024, 768))
		                                  .with_title("Colorful Game Of Life")
		                                  .build_vk_surface(&event_loop, renderer.instance.clone())?;
		
		let window = surface.window();
		let size = window.outer_size();
		let monitor_size = window.current_monitor().size();
		window.set_outer_position(PhysicalPosition::new((monitor_size.width - size.width) / 2, (monitor_size.height - size.height) / 2));
		
		renderer.create_swapchain(surface)?;
		
		Ok(Application {
			renderer,
			event_loop,
		})
	}
	
	pub fn run(self) -> ! {
		let mut renderer = self.renderer;
		let event_loop = self.event_loop;
		
		event_loop.run(move |event, _, control_flow| {
			*control_flow = ControlFlow::Poll;
			
			match event {
				Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
					*control_flow = ControlFlow::Exit;
				},
				
				Event::WindowEvent {
					event: WindowEvent::KeyboardInput {
						input: KeyboardInput {
							virtual_keycode: Some(VirtualKeyCode::R),
							state: ElementState::Pressed, ..
						}, ..
					}, ..
				} => {
					renderer.regenerate();
				},
				
				Event::WindowEvent {
					event: WindowEvent::KeyboardInput {
						input: KeyboardInput {
							virtual_keycode: Some(VirtualKeyCode::Q),
							state: ElementState::Pressed, ..
						}, ..
					}, ..
				} => {
					*control_flow = ControlFlow::Exit;
				},
				
				Event::WindowEvent {
					event: WindowEvent::KeyboardInput {
						input: KeyboardInput {
							virtual_keycode: Some(VirtualKeyCode::F),
							state: ElementState::Pressed, ..
						}, ..
					}, ..
				} => {
					let window = renderer.surface.as_ref().unwrap().window();
					
					if let None = window.fullscreen() {
						window.set_fullscreen(Some(Fullscreen::Borderless(window.current_monitor())));
						window.set_cursor_visible(false);
					} else {
						window.set_fullscreen(None);
						window.set_cursor_visible(true);
					}
				},
				
				Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
					renderer.recreate_swapchain();
				},
				
				Event::RedrawRequested(_) | Event::RedrawEventsCleared => {
					loop {
						match renderer.render() {
							Ok(()) => break,
							Err(RenderError::Retry) => continue,
							Err(err) => {
								eprintln!("{}", err);
								*control_flow = ControlFlow::Exit;
							}
						}
						break;
					}
				},
				_ => {},
			}
		})
	}
}
