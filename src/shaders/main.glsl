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
}
