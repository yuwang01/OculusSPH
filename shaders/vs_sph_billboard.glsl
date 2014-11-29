#version 400

in vec4 vertex_position;

uniform mat4 view, proj;

out vec3 Color;

void main() 
{ 
//	gl_Position = mvp * vec4(vp.x, vp.y, vp.z, 1.0);
//	gl_Position = vec4(vp.x, vp.y, vp.z, 1.0);
	gl_Position = proj * view * vec4(vertex_position.xyz, 1.0);
}
