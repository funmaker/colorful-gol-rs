use std::error::Error;
use std::env;

use getopts::Options;

mod shaders;
mod renderer;
mod application;

use renderer::Renderer;
use application::Application;

fn main() -> Result<(), Box<dyn Error>> {
	let args: Vec<String> = env::args().collect();
	let program = args[0].clone();
	let mut opts = Options::new();
	
	opts.optopt("d", "device", "Select device to use", "NUMBER");
	opts.optflag("", "debug", "Enable debugging layer and info");
	opts.optflag("h", "help", "Print this help menu");
	
	let matches = opts.parse(&args[1..])?;
	
	if matches.opt_present("h") {
		print_usage(&program, opts);
		return Ok(());
	}
	
	let device = matches.opt_get("d")?;
	let debug = matches.opt_present("debug");
	
	let renderer = Renderer::new(device, debug)?;
	let application = Application::new(renderer)?;
	
	application.run()
}

fn print_usage(program: &str, opts: Options) {
	let brief = format!("Usage: {} [options]", program);
	print!("{}", opts.usage(&brief));
}
