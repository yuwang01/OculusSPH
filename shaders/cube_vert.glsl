#version 400

in vec3 vertex_position;

uniform mat4 view, proj;

//uniform vec3 cube;
//uniform vec3 vNormal;

out vec4 position;

//out vec3 light;
//out vec3 normal;
//out vec3 eye;
    
void main()
{
    // vec3 lightPos = vec3(0.0, 0.0, 3.0);
    // vec3 eyePos = vec3(0.0, 0.0, 0.0);

    // vec4 N = normalize(np * vec4(vNormal, 1.0));

    // normal = N.xyz;

    // light = lightPos - position.xyz;
    // eye = eyePos - position.xyz;

    gl_Position = proj * view * vec4(vertex_position.xyz, 1.0);
        
}