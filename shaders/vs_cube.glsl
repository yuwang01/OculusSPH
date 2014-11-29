#version 400

in vec3 vp;
uniform mat4 proj, view;
out vec3 texcoords;

void main () {
	texcoords = vp;
	
	gl_Position = proj * view * vec4 (vp, 1.0);
}