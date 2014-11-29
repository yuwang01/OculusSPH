#version 400

//varying vec4 position;
//varying vec3 light;
//varying vec3 normal;

//varying vec3 eye;

out vec4 frag_color;

void main() {
        
    // vec3 nNormal = normalize(normal);
    // vec3 nVertToLight = normalize(light);
    // vec3 nVertToEye = normalize(eye);

    // float ambientTerm = 0.02;
    // vec3 ambientColor = vec3(.4, .4, .4);
    // vec3 Iamb = ambientColor * ambientTerm;

    // float dotVL = clamp(dot(nNormal, nVertToLight), 0.0, 1.0);
    // float diffuseTerm = .3;
    
    // // vec3 diffuseColor = vec3(0.74, 0.71, 0.42);
    // vec3 diffuseColor = vec3(0.6, 0.6, 0.6);
    // vec3 Idiff = diffuseColor * diffuseTerm * dotVL;
        
    // float shinness = 100.;
    // float specularTerm = 1.0 - ambientTerm - diffuseTerm;
    // vec3 specularColor = vec3(1.0, 1.0, 1.0);
        
    // vec3 H = normalize(nVertToEye + nVertToLight);
    // float dotNH = clamp(dot(nNormal, H), 0.0, 1.0);

    // vec3 Ispec = specularColor * specularTerm * pow(dotNH, shinness);

    // vec3 color = Iamb + Idiff + Ispec;

    // gl_FragColor = vec4(color, 1.0);

    frag_color = vec4(1.0, 0.0, 0.0, 1.0);
}