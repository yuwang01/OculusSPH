#version 400

layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 normal;

uniform mat4 view, proj;

out vec3 position_eye, normal_eye;

void main() 
{ 
	position_eye = vec3(view * vec4(vertex_position, 1.0));
	normal_eye = vec3(view * vec4(normal, 0.0));
	// normal_eye = vec3(view * vec4(vertex_position, 0.0));

	gl_Position = proj * view * vec4(vertex_position.xyz, 1.0);
}