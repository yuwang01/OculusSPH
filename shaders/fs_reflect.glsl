#version 400

in vec3 position_eye;
in vec3 normal_eye;

uniform samplerCube cube_texture;
uniform mat4 view;

out vec4 frag_colour;

void main () {
	/* reflect ray around normal from eye to surface */
	vec3 incident_eye = normalize (position_eye);
	vec3 normal = normalize (normal_eye);

	vec3 reflected = reflect (incident_eye, normal);
	// convert from eye to world space
	reflected = vec3 (inverse (view) * vec4 (reflected, 0.0));

	frag_colour = texture (cube_texture, reflected);
}