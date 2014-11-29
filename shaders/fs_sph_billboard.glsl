#version 400

out vec4 frag_color;

void main() 
{
	vec4 InnerColor = vec4(.6, .2, .10, 1.0);
	vec4 OuterColor = vec4(0.0, 0.0, 0.0, 0.0);

	float dx = (gl_PointCoord.x - 0.5);
	float dy = (gl_PointCoord.y - 0.5);
        
    float r = sqrt(dx*dx + dy*dy);
    float r1 = 0.1;

    frag_color = vec4(mix(InnerColor, OuterColor, smoothstep(r1, 1.0, r)));
}