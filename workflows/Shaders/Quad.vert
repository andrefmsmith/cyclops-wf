#version 400
uniform mat4 transform = mat4(1);
layout(location = 0) in vec2 vp;
layout(location = 1) in vec2 vt;
out vec2 texCoord;

void main()
{
  gl_Position = vec4(vp.x * 2 + transform[3][0], vp.y * 2 * 1.6, -1.0, 1.0);
  texCoord = (mat3(transform) * vec3(vt - vec2(0.5,0.5), 0)).xy + vec2(0.5,0.5);
}
