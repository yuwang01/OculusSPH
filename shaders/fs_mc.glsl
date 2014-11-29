#version 400

in vec3 position_eye_, normal_eye_;

uniform mat4 view;

vec3 surfaceColor = vec3(0.8, 0.1, 0.1);

// fixed point light properties
vec3 light_position_world  = vec3 (5.0, 5.0, 10.0);
vec3 Ls = vec3 (0.2, 0.2, 0.2);
vec3 Ld = vec3 (0.9, 0.9, 0.9);
vec3 La = vec3 (0.3, 0.3, 0.3);

// surface reflectance
vec3 Ks = vec3 (0.5, 0.5, 0.5); // fully reflect specular light
vec3 Kd = vec3 (0.5, 0.5, 0.5); // orange diffuse surface reflectance
vec3 Ka = vec3 (0.2, 0.2, 0.2); // fully reflect ambient light
float specular_exponent = 100.0; // specular 'power'

out vec4 frag_color;

void main()
{
	// ambient intensity
	vec3 Ia = La * Ka;

	// vec3 normal = vec3(normal_eye.x+.5, normal_eye.y+.5, normal_eye.z+.5);

	// diffuse intensity
	// raise light position to eye space
	vec3 light_position_eye = vec3 (view * vec4(light_position_world, 1.0));
	vec3 distance_to_light_eye = light_position_eye - position_eye_;
	vec3 direction_to_light_eye = normalize (distance_to_light_eye);
	float dot_prod = dot (direction_to_light_eye, normalize(normal_eye_));
	dot_prod = max (dot_prod, 0.0);
	vec3 Id = Ld * Kd * dot_prod * surfaceColor; // final diffuse intensity
	
	// specular intensity
	vec3 surface_to_viewer_eye = normalize (-position_eye_);
	
	// vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
	// float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
	// dot_prod_specular = max (dot_prod_specular, 0.0);
	// float specular_factor = pow (dot_prod_specular, specular_exponent);
	
	// blinn
	vec3 half_way_eye = normalize (surface_to_viewer_eye + direction_to_light_eye);
	float dot_prod_specular = max (dot (half_way_eye, normalize(normal_eye_)), 0.0);
	float specular_factor = pow (dot_prod_specular, specular_exponent);
	
	vec3 Is = Ls * Ks * specular_factor * dot_prod; // final specular intensity
	
	// final colour
	frag_color = vec4 (Is + Id + Ia, 1.0);
	
	// frag_color = vec4(dot_prod, dot_prod, dot_prod, 1.0);
	// frag_color = vec4((normalize(normal_eye).x+1.0)/2, (normalize(normal_eye).y+1.0)/2, (normalize(normal_eye).z+1.0)/2, 1.0);

}
