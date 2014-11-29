#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "maths_funcs.h"

#include "stb_image.h"
#include "maths_funcs.h"
#include "help.h"
#include "stats.h"

#ifdef WIN32
#define OVR_OS_WIN32
#elif defined(__APPLE__)
#define OVR_OS_MAC
#else
#define OVR_OS_LINUX
#include <X11/Xlib.h>
#include <GL/glx.h>
#endif

#include "help.h"

#include "../OculusSDK/LibOVR/Include/OVR.h"
#include "../OculusSDK/LibOVR/Src/OVR_CAPI.h"
#include "../OculusSDK/LibOVR/Src/OVR_CAPI_GL.h"
#include "../OculusSDK/LibOVR/Src/CAPI/CAPI_HSWDisplay.h"

#define USE_GL_ATTACHMENTS                      (1)  // enable OpenGL attachments for Compute results

#define COMPUTE_KERNEL_FILENAME                 ("kernels.cl")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_applyBodyForce   ("sph_kernel_applyBodyForce")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_advance          ("sph_kernel_advance")
// #define COMPUTE_KERNEL_METHOD_NAME_SPH_hashparticles    ("sph_kernel_hashparticles")
// #define COMPUTE_KERNEL_METHOD_NAME_SPH_sort             ("sph_kernel_sort")
// #define COMPUTE_KERNEL_METHOD_NAME_SPH_sortPostPass     ("sph_kernel_sortPostPass")
// #define COMPUTE_KERNEL_METHOD_NAME_COPY                 ("kernel_copy")
// #define COMPUTE_KERNEL_METHOD_NAME_SPH_indexx           ("sph_kernel_indexx")
// #define COMPUTE_KERNEL_METHOD_NAME_SPH_indexPostPass    ("sph_kernel_indexPostPass")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_findNeighbors    ("sph_kernel_findNeighbors")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_pressure         ("sph_kernel_pressure")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_calcRelaxPos     ("sph_kernel_calcRelaxPos")
#define COMPUTE_KERNEL_METHOD_NAME_SPH_moveToRelaxPos   ("sph_kernel_moveToRelaxPos")
#define COMPUTE_KERNEL_METHOD_NAME_resolveCollisions    ("sph_kernel_resolveCollisions")
#define COMPUTE_KERNEL_METHOD_NAME_MC_RESET             ("mc_kernel_reset")
#define COMPUTE_KERNEL_METHOD_NAME_MC_gridval           ("mc_kernel_gridval")
#define COMPUTE_KERNEL_METHOD_NAME_MC_gridGrad          ("mc_kernel_gridGrad")
#define COMPUTE_KERNEL_METHOD_NAME_MC_CUBEINDEX         ("mc_kernel_cubeindex")
#define COMPUTE_KERNEL_METHOD_NAME_MC_INTERPOLATE       ("mc_kernel_interpolate")

#define ABS(x) (x < 0 ? -(x) : (x))

////////////////////////////////////////////////////////////////////////////////////////////////

static cl_context ComputeContext;

static cl_kernel ComputeKernel_sph_kernel_applyBodyForce;
static cl_kernel ComputeKernel_sph_kernel_advance;
// static cl_kernel ComputeKernel_sph_kernel_hashparticles;
// static cl_kernel ComputeKernel_sph_kernel_sort;
// static cl_kernel ComputeKernel_sph_kernel_sortPostPass;
// static cl_kernel ComputeKernel_copy;
// static cl_kernel ComputeKernel_sph_kernel_indexx;
// static cl_kernel ComputeKernel_sph_kernel_indexPostPass;
static cl_kernel ComputeKernel_sph_kernel_findNeighbors;
static cl_kernel ComputeKernel_sph_kernel_pressure;
static cl_kernel ComputeKernel_sph_kernel_calcRelaxPos;
static cl_kernel ComputeKernel_sph_kernel_moveToRelaxPos;
static cl_kernel ComputeKernel_sph_kernel_resolveCollisions;

static cl_kernel ComputeKernel_mc_kernel_reset;
static cl_kernel ComputeKernel_mc_kernel_gridval;
static cl_kernel ComputeKernel_mc_kernel_gridGrad;
static cl_kernel ComputeKernel_mc_kernel_cubeindex;
static cl_kernel ComputeKernel_mc_kernel_vertInterp;
// static cl_kernel ComputeKernel_mc_kernel_polygonisecube;
// static cl_kernel ComputeKernel_mc_kernel_packtriangles;

static cl_program ComputeProgram;
static cl_device_id ComputeDeviceId;
static cl_command_queue ComputeCommands;
static cl_device_type ComputeDeviceType;
static CGLContextObj OpenGLCotext;

static bool isrunning = false;
static bool renderPart = true;

static bool simpleTri = false;
static bool blinn = true;
static bool reflect = false;
static bool refract = false;

static bool normalFlat = false;
static bool normalSmooth = true;

static bool t_was_down = false;
static bool g_was_down = false;
static bool y_was_down = false;
static bool h_was_down = false;

static int initOption = 2;
static int numFrames = 0;
static bool dump_video = false;
static double video_dump_timer = 0.0;

static unsigned char* g_video_memory_start = NULL;
static unsigned char* g_video_memory_ptr = NULL;
static int g_video_seconds_total = 20;
static int g_video_fps = 25;

static float* position;
static float* velocity;
static float* density;
static int* partIdx;

static bool cam_moved = true;
static int button = -1;
static int oldbutton = -1;

static double oldX = 0.0;
static double oldY = 0.0;

static float* gridCellIndex; // gridCellIndex is of size sizeof(float4) * (Nx+1) * (Ny+1) * (Nz+1)
static int* gridPointIndex; // gridPointIndex is of size 8 * sizeof(int) * (Nx) * (Ny) * (Nz)
static int* cubeIndex;
static float* gridCellGrad;

static TRIANGLE *tri;
static TRIANGLE *norm;
static TRIANGLE *normFlat;

static int *triIdx;

static TRIANGLE *triFull;
static TRIANGLE *triFullCopy;

static TRIANGLE *normFull;
static TRIANGLE *normFullCopy;

static int* triCount;

static FILE* statistics;

static mat4 rotation;

static unsigned int POS_ATTRIB_SIZE           = 4;
static unsigned int VEL_ATTRIB_SIZE           = 4;
static unsigned int PARTICLEIndex_ATTRIB_SIZE = 2;
static unsigned int GRIDCELLIndex_ATTRIB_SIZE = 1;
static unsigned int NEIGHBOR_MAP_ATTRIB_SIZE  = 128;
static unsigned int DENSITY_ATTRIB_SIZE       = 1; 
static unsigned int PRESSURE_ATTRIB_SIZE      = 1;

static unsigned int GRIDCELL_ATTRIB_SIZE      = 4;
static unsigned int GRIDPOINT_ATTRIB_SIZE     = 8;

static unsigned int GRIDCELL_GRAD_ATTRIB_SIZE = 4;
static unsigned int TRI_ATTRIB_SIZE           = 9;

static GLuint vao;
static GLuint position_vbo;

static GLuint vaoTri;
static GLuint triangle_vbo;

static GLuint vaoBox;
static GLuint box_vbo;

static GLuint vaoNorm;
static GLuint norm_vbo;

static int outer_tess_fac_loc;
static int inner_tess_fac_loc;

static size_t SPHPosbufferSize;
static size_t SPHVelbufferSize;
static size_t SPHNeighborbufferSize;
static size_t SPHDensitybufferSize;
static size_t SPHPressurebufferSize;

static size_t MCGridCellbufferSize;
static size_t MCGridCellIndexbufferSize;
static size_t MCGridGradbufferSize;
static size_t MCTriBufferSize;
static size_t MCTriCountBufferSize;

static unsigned int shader_program;
static unsigned int triangle_program;
static unsigned int cube_program;
static unsigned int reflect_program;
static unsigned int refract_program;

static cl_mem PositionBuffer;
static cl_mem VelocityBuffer;
static cl_mem prevPositionBuffer;
static cl_mem neighborMapBuffer;
static cl_mem densityBuffer;
static cl_mem nearDensityBuffer;
static cl_mem pressureBuffer;
static cl_mem nearPressureBuffer;
static cl_mem RelaxedPosBuffer;
static cl_mem MCGridBuffer;
static cl_mem MCGridGradBuffer;
static cl_mem MCEdgeTableBuffer;
static cl_mem MCTriTableBuffer;
static cl_mem MCCubeIndexBuffer; // cubeindex for each cube
static cl_mem MCCubePointIndexBuffer; // 8 indices for each cube to be used in mcgrid
static cl_mem MCTriBuffer;
static cl_mem MCNormBuffer;
static cl_mem MCTriCountBuffer;

#if (USE_GL_ATTACHMENTS)
static GLenum BufferModeType                    = GL_STATIC_DRAW_ARB;
#else
static GLenum BufferModeType                    = GL_DYNAMIC_DRAW_ARB;
#endif

static bool debug = false;
static GLFWwindow* window;
static ovrHmd hmd;
static ovrSizei eyeres[2];
static ovrEyeRenderDesc eye_rdesc[2];
static ovrGLTexture fb_ovr_tex[2];
static union ovrGLConfig glcfg;

static unsigned int hmd_caps;
static unsigned int distort_caps;

static int fb_width, fb_height;
static unsigned int fbo, fb_tex, fb_depth;
static int fb_tex_width, fb_tex_height;

static unsigned int hellotri_program;

static void usage();
static int arrayIndexFromCoordinate(int i, int xSize, int j, int ySize, int k);
static int run_sph_kernel_applyBodyForce();
static int run_sph_kernel_advance();
static int run_sph_kernel_findNeighbors();
static int run_sph_kernel_pressure();
static int run_sph_kernel_calcRelaxPos();
static int run_sph_kernel_moveToRelaxPos();
static int run_sph_kernel_resolveCollisions();
static int run_mc_kernel_reset();
static int run_mc_kernel_gridval();
static int run_mc_kernel_gridGrad();
static int run_mc_kernel_cubeindex();
static int run_mc_kernel_interpolate();
static void shutdown_opencl();
static int init(int gpu);
static int initGLVR(void);
static int run();
static int setup_compute_devices(int gpu);
static int setup_compute_memory();
static int setup_compute_kernels(void);
static int setup_opencl(int use_gpu);
static void InitSPHParticle(int idx, float px, float py, float pz, float vx, float vy, float vz);
static void InitBreakDam();
static void InitMidAirDrop();
static void InitTwoCubes();
static void InitBallDrop();
static void InitParticleState(int initOption);
static void InitGridCellState();
static void cleanup(void);
static void reshape(int, int);
static void update_rtarg(int, int);
static unsigned int next_pow2(unsigned int);
static void quat_to_matrix(const float*, float*);
static void KeyCallback(GLFWwindow*, int, int, int, int);

/* forward declaration to avoid including non-public headers of libovr */
OVR_EXPORT void ovrhmd_EnableHSWDisplaySDKRender(ovrHmd hmd, ovrBool enable);

int main(int argc, char **argv)
{   
    usage();

    if (argc == 2)
    {
        initOption = atoi(argv[1]);
    }

    if (argc == 5)
    {
        initOption = atoi(argv[1]);
        volNx = atoi(argv[2]);
        volNy = atoi(argv[3]);
        volNz = atoi(argv[4]);
    }
    else {
        volNx = 31;
        volNy = 31;
        volNz = 31;
    }

    printf("volNx = %d\tvolNy = %d\tvolNz = %d\n", volNx, volNy, volNz);

    maxTri = 5 * volNx * volNy * volNz;

    volInd = (volNx + 1) * (volNy + 1) * (volNz + 1);

    nvNx = volNx + 1;
    nvNy = volNy + 1;
    nvNz = volNz + 1;
    
    int use_gpu = 1;

    statistics = fopen("statistics.log", "wb");
    if (!statistics)
    {
        printf("Cannot open file statistics.log\n");
        exit(1);
    }

    kScreenWidth = g_gl_width;
    kScreenHeight = g_gl_height;
    kViewHeight = kScreenHeight*kViewWidth/kScreenWidth;
    kH = 6*kParticleRadius;
    kCellSize = kH;
    
    printf("kCellSize = %f\n", kCellSize);

    kGridWidth = ceil(kViewWidth/kCellSize);
    kGridHeight = ceil(kViewHeight/kCellSize);
    kGridDepth = ceil(kViewDepth/kCellSize);

    Nx = kGridWidth;
    Ny = kGridHeight;
    Nz = kGridDepth;

    printf("kGridWidth = %f\tkGridHeight = %f\tkGridDepth = %f\n", kGridWidth, kGridHeight, kGridDepth);
    
    // kGridCellCount = kGridWidth * kGridHeight * kGridDepth;

    voxel = Nx * Ny * Nz;

    volEdgeX = kViewWidth / volNx;
    volEdgeY = kViewHeight / volNy;
    volEdgeZ = kViewDepth / volNz;

    printf("volEdgeX = %f\tvolEdgeY = %f\tvolEdgeZ = %f\n", volEdgeX, volEdgeY, volEdgeZ);

    NEIGHBOR_MAP_ATTRIB_SIZE = kMaxNeighbourCount;

    kDt = (1.0/kFrameRate) / kSubSteps;
    kDt2 = kDt*kDt;
    kNorm = 20/(2*kPi*kH*kH);
    kNearNorm = 30/(2*kPi*kH*kH);
    kEpsilon2 = kEpsilon*kEpsilon;

    int eyeIndex;
    ovrPosef pose[2];

    cam_pos[0] = 0.0f;
    cam_pos[1] = 0.0f;
    cam_pos[2] = 1.2*kViewDepth;

    position = (float*)malloc(sizeof(float) * POS_ATTRIB_SIZE * kParticleCount);
    velocity = (float*)malloc(sizeof(float) * VEL_ATTRIB_SIZE * kParticleCount);
    density = (float*)malloc(sizeof(float) * kParticleCount);

    tri = (TRIANGLE*)malloc(maxTri*sizeof(float)*9);
    norm = (TRIANGLE*)malloc(maxTri*sizeof(float)*9);
    normFlat = (TRIANGLE*)malloc(maxTri*sizeof(float)*9);

    triIdx = (int*)malloc(maxTri*sizeof(int)*3);

    triFull = (TRIANGLE*)malloc(maxTri * sizeof(float)*9);
    triFullCopy = (TRIANGLE*)malloc(maxTri * sizeof(TRIANGLE));
    triCount = (int*)malloc(sizeof(int)*volNx*volNy*volNz);

    normFull = (TRIANGLE*)malloc(maxTri * sizeof(float)*9);
    normFullCopy = (TRIANGLE*)malloc(maxTri * sizeof(float)*9);

    // partIdx = (int*)malloc(sizeof(int) * PARTICLEIndex_ATTRIB_SIZE * kParticleCount);
    memset(position, 0.0, sizeof(float) * POS_ATTRIB_SIZE * kParticleCount);
    memset(velocity, 0.0, sizeof(float) * VEL_ATTRIB_SIZE * kParticleCount);
    // memset(partIdx, 0.0, sizeof(int) * PARTICLEIndex_ATTRIB_SIZE * kParticleCount);
    memset(tri, 0.0, sizeof(float) * 9 * maxTri);
    memset(norm, 0.0, sizeof(float) * 9 * maxTri);
    memset(normFlat, 0.0, sizeof(float) * 9 * maxTri);

    memset(triFull, 0.0, sizeof(float)*9*maxTri);
    memset(triFullCopy, 0.0, sizeof(float)*9*maxTri);
    memset(triCount, 0, sizeof(int)*volNx*volNy*volNz);

    memset(normFull, 0.0, sizeof(float)*9*maxTri);
    memset(normFullCopy, 0.0, sizeof(float)*9*maxTri);

    gridCellIndex = (float*)malloc(sizeof(float) * GRIDCELL_ATTRIB_SIZE * (volNx + 1) * (volNy + 1) * (volNz + 1));
    gridPointIndex = (int*)malloc(sizeof(int) * GRIDPOINT_ATTRIB_SIZE * volNx * volNy * volNz);
    gridCellGrad = (float*)malloc(sizeof(float) * GRIDCELL_GRAD_ATTRIB_SIZE * (volNx+1) * (volNy+1) * (volNz+1));

    printf("memsetting gridCellIndex with %d floats\n", GRIDCELL_ATTRIB_SIZE * (volNx + 1) * (volNy + 1) * (volNz + 1));
    memset(gridCellIndex, 0.0, sizeof(float) * GRIDCELL_ATTRIB_SIZE * (volNx + 1) * (volNy + 1) * (volNz + 1));

    printf("memsetting gridPointIndex with %d integers\n", GRIDPOINT_ATTRIB_SIZE * volNx * volNy * volNz);
    memset(gridPointIndex, 0, sizeof(int) * GRIDPOINT_ATTRIB_SIZE * volNx * volNy * volNz);

    printf("memsetting gridCellGrad with %d floats\n", (volNx+1)*(volNy+1)*(volNz+1));
    memset(gridCellGrad, 0.0, sizeof(float) * GRIDCELL_GRAD_ATTRIB_SIZE * (volNx+1)*(volNy+1)*(volNz+1));

    int i, j, k;

    for (k = 0; k < volNz+1; k++)
    {
        for (j = 0; j < volNy+1; j++)
        {
            for (i = 0; i < volNx+1; i++)
            {
                if ((i == 0) || (i == volNx) || (j == 0) || (j == volNy) || (k == 0) || (k == volNz))
                    gridCellGrad[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+3] = -1.0;
            }
        }
    }

    cubeIndex = (int*)malloc(sizeof(int) * volNx * volNy * volNz);
    printf("memsetting cubeIndex with %d ints\n", volNx * volNy * volNz);
    memset(cubeIndex, 0, sizeof(unsigned char) * volNx * volNy * volNz);

    previous_seconds = glfwGetTime();
    frame_count = 0;
    
    // initialise timers 
    double video_timer = 0.0;
    
    // time video has been recording 
    double video_dump_timer = 0.0; // timer for next frame grab 
    double frame_time = 0.04; // 1/25 seconds of time

    if (init(use_gpu) == GL_NO_ERROR)
    {
        while (!glfwWindowShouldClose(window))
        {
            static double previousseconds = glfwGetTime();
            double currentseconds = glfwGetTime ();
            elapsedseconds = currentseconds - previousseconds;
            previousseconds = currentseconds;

            fps(window);
            
            if (isrunning == true)
                run();
            
            glBindBuffer(GL_ARRAY_BUFFER, triangle_vbo);
            glBufferData(GL_ARRAY_BUFFER, maxTri * sizeof(TRIANGLE), tri, GL_DYNAMIC_DRAW);

            // the drawing starts with a call to ovrHmd_BeginFrame 
            ovrHmd_BeginFrame(hmd, 0);

            // start drawing onto our texture render target 
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            //for each eye ...
            for(eyeIndex = 0; eyeIndex < ovrEye_Count; eyeIndex++) 
            {
                ovrEyeType eye = hmd->EyeRenderOrder[eyeIndex];
                pose[eye] = ovrHmd_GetHmdPosePerEye(hmd, eye);
                
                OVR::Matrix4f l_ProjectionMatrix = ovrMatrix4f_Projection(hmd->DefaultEyeFov[eye], 0.5, 500.0, 1);
                OVR::Quatf l_Orientation = OVR::Quatf(pose[eye].Orientation);
                OVR::Matrix4f l_Translation = OVR::Matrix4f::Translation(-1*cam_pos[0], -1*cam_pos[1], -1*cam_pos[2]);
                OVR::Matrix4f l_ModelViewMatrix = OVR::Matrix4f(l_Orientation.Inverted()) * l_Translation;

                glViewport(eye == ovrEye_Left ? 0 : fb_width / 2, 0, fb_width / 2, fb_height);
                
                glUseProgram(hellotri_program);
                glUniformMatrix4fv(tri_proj_mat_location, 1, GL_FALSE, &(l_ProjectionMatrix.Transposed().M[0][0]));
                
                glUseProgram(shader_program);
                glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, &(l_ProjectionMatrix.Transposed().M[0][0]));

                glUseProgram(triangle_program);
                glUniformMatrix4fv(mc_proj_mat_location, 1, GL_FALSE, &(l_ProjectionMatrix.Transposed().M[0][0]));

                if (cam_moved == true)
                {
                    OVR::Matrix4f l_Translation = OVR::Matrix4f::Translation(-1*cam_pos[0], -1*cam_pos[1], -1*cam_pos[2]);
                    OVR::Matrix4f l_ModelViewMatrix = OVR::Matrix4f(l_Orientation.Inverted()) * l_Translation;
                    
                    glUseProgram(hellotri_program);
                    glUniformMatrix4fv(tri_view_mat_location, 1, GL_FALSE, &(l_ModelViewMatrix.Transposed().M[0][0]));

                    glUseProgram(shader_program);
                    glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, &(l_ModelViewMatrix.Transposed().M[0][0]));

                    glUseProgram(triangle_program);
                    glUniformMatrix4fv(mc_view_mat_location, 1, GL_FALSE, &(l_ModelViewMatrix.Transposed().M[0][0]));
                }

                if (simpleTri == true)
                {
                    // draw test triangle
                    glUseProgram(hellotri_program);
                    glBindVertexArray(vertexArray);
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                }

                if (renderPart == true)
                {
                    // draw particles
                    glUseProgram(shader_program);            
                    glBindVertexArray(vao);
                    glPointSize(20.0);
                    glDrawArrays (GL_POINTS, 0, kParticleCount);
                }

                if (normalFlat == true)
                {
                    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo);
                    glBufferData(GL_ARRAY_BUFFER, maxTri * sizeof(TRIANGLE), normFlat, GL_DYNAMIC_DRAW);
                }

                if (normalFlat == false)
                {
                    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo);
                    glBufferData(GL_ARRAY_BUFFER, maxTri * sizeof(TRIANGLE), norm, GL_DYNAMIC_DRAW);
                }

                if (blinn == true)
                {
                    // draw mesh
                    glUseProgram(triangle_program);
                    glBindVertexArray(vaoTri);
                    glPatchParameteri (GL_PATCH_VERTICES, 3);
                    glDrawArrays (GL_PATCHES, 0, Ntri*9);
                    // glDrawArrays(GL_LINE, 0, Ntri*9);
                } 
            }
            
            ovrHmd_EndFrame(hmd, pose, &fb_ovr_tex[0].Texture);

            glfwPollEvents();

            // glfwSwapBuffers(window);

        }
    }

    glfwTerminate();
    
    return 0;
}

void printMesh()
{
    FILE* fp;
    fp = fopen("Mesh.ply", "w");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file Mesh.ply\n");
        exit(-1);
    }

    fprintf(fp, "ply\nformat ascii 1.0\nelement vertex %d\nproperty float32 x\nproperty float32 y\nproperty float32 z\nelement face %d\nproperty list uint8 int32 vertex_indices\nend_header\n", Ntri * 3, Ntri);

    int i, k;

    for (i=0;i<Ntri;i++) {
      // fprintf(fptr,"f3 ");
      for (k=0;k<3;k++)  {
         fprintf(fp,"%g %g %g\n",tri[i].p[k].x,tri[i].p[k].y,tri[i].p[k].z);
      }
        // fprintf(fptr," 0.5 0.5 0.5\n"); // colour
    }

   for (i = 0; i < Ntri; i++)
   {
      fprintf(fp, "3 %d %d %d\n", 3*i, 3*i+1, 3*i+2);
   }

    fclose(fp);
}

void usage()
{
    printf("Usage:\n");
    printf("\t\tSpace - Start Sim\n");
    printf("\t\tEnter - Stop Sim\n");
    printf("\t\t1 - Blinn Phong shading\n");
    printf("\t\t2 - Reflection\n");
    printf("\t\t3 - Refraction\n");
    printf("\t\tL - Wireframe\n");
    printf("\t\tK - Wireframe Off\n");
    printf("\t\tP - Display particles\n");
    printf("\t\tO - Particle Off\n");
    printf("\t\tW - Move forwards\n");
    printf("\t\tS - Move backwards\n");
    printf("\t\tA - Move left\n");
    printf("\t\tD - Move right\n");
    printf("\t\tB - Smooth normal\n");
    printf("\t\tN - Face normal\n");
    printf("\t\tArrow left - rotate to left\n");
    printf("\t\tArrow right - rotate to right\n");
    printf("\t\tPage up - move up\n");
    printf("\t\tPage down - move down\n");
    printf("\t\tF - Write position to file\n");
    printf("\t\tM - Write mesh to file\n");
    printf("\t\tV - Recored video frames\n");
    printf("\t\tEsc - exit\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////
int arrayIndexFromCoordinate(int i, int xSize, int j, int ySize, int k)
{
    return i + j * xSize + k * xSize * ySize;
}

void shutdown_opencl()
{
    clFinish(ComputeCommands);

    clReleaseMemObject(VelocityBuffer);
    // clReleaseMemObject(SortedVelocityBuffer);
    clReleaseMemObject(PositionBuffer);
    // clReleaseMemObject(SortedPositionBuffer);
    clReleaseMemObject(prevPositionBuffer);
    // clReleaseMemObject(SortedPrevPositionBuffer);
    // clReleaseMemObject(particleIndex);
    // clReleaseMemObject(gridCellIndexBuffer);
    // clReleaseMemObject(gridCellIndexFixedUpBuffer);
    clReleaseMemObject(neighborMapBuffer);
    clReleaseMemObject(densityBuffer);
    clReleaseMemObject(nearDensityBuffer);
    clReleaseMemObject(pressureBuffer);
    clReleaseMemObject(nearPressureBuffer);
    clReleaseMemObject(MCGridBuffer);
    clReleaseMemObject(MCGridGradBuffer);
    clReleaseMemObject(MCEdgeTableBuffer);
    clReleaseMemObject(MCTriTableBuffer);
    clReleaseMemObject(MCCubeIndexBuffer);
    clReleaseMemObject(MCCubePointIndexBuffer);
    clReleaseMemObject(MCTriBuffer);
    clReleaseMemObject(MCTriCountBuffer);
    clReleaseMemObject(MCNormBuffer);

    clReleaseKernel(ComputeKernel_sph_kernel_applyBodyForce);
    clReleaseKernel(ComputeKernel_sph_kernel_advance);
    // clReleaseKernel(ComputeKernel_sph_kernel_hashparticles);
    // clReleaseKernel(ComputeKernel_sph_kernel_sort);
    // clReleaseKernel(ComputeKernel_sph_kernel_sortPostPass);
    // clReleaseKernel(ComputeKernel_copy);
    // clReleaseKernel(ComputeKernel_sph_kernel_indexx);
    // clReleaseKernel(ComputeKernel_sph_kernel_indexPostPass);
    clReleaseKernel(ComputeKernel_sph_kernel_findNeighbors);
    clReleaseKernel(ComputeKernel_sph_kernel_pressure);
    clReleaseKernel(ComputeKernel_sph_kernel_calcRelaxPos);
    clReleaseKernel(ComputeKernel_sph_kernel_moveToRelaxPos);
    clReleaseKernel(ComputeKernel_sph_kernel_resolveCollisions);
    clReleaseKernel(ComputeKernel_mc_kernel_reset);
    clReleaseKernel(ComputeKernel_mc_kernel_gridval);
    clReleaseKernel(ComputeKernel_mc_kernel_gridGrad);
    clReleaseKernel(ComputeKernel_mc_kernel_cubeindex);
    clReleaseKernel(ComputeKernel_mc_kernel_vertInterp);

    clReleaseProgram(ComputeProgram);
    clReleaseContext(ComputeContext);

    if (position)
        free(position);
    if (velocity)
        free(velocity);
    if (gridCellIndex)
        free(gridCellIndex);
    if (gridCellGrad)
        free(gridCellGrad);
    if (gridPointIndex)
        free(gridPointIndex);
    if (cubeIndex)
        free(cubeIndex);
    if (tri)
        free(tri);
    if (norm)
        free(norm);
    if (normFlat)
        free(normFlat);
    if (triFull)
        free(triFull);
    if (triFullCopy)
        free(triFullCopy);
}

int init(int gpu)
{
    /*------------------------setup opengl and opencl------------------------*/
    if (initGLVR() == -1)
    {
        fprintf(stderr, "Error: initGLVR()\n");
        return EXIT_FAILURE;
    }

    int err;
    
    err = setup_opencl(gpu);
    if (err != GL_NO_ERROR)
    {
        printf("Failed to setup OpenCL state! Error %d\n", err);
        exit(err);
    }
    
    return CL_SUCCESS;
}

int initGLVR(void)
{
    fprintf(stdout, "%s\n", SEPARATOR);

    int i, x, y;
    unsigned int flags;

    /* libovr must be initialized before we create the OpenGL context */
    ovr_Initialize();

    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        return 1;
    }
    
    char message[256];
    sprintf(message, "starting GLFW %s", glfwGetVersionString());
    assert(gl_log(message, __FILE__, __LINE__));
    glfwSetErrorCallback(glfw_error_callback);

    // uncomment these lines if on Apple OS X
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    glfwWindowHint(GLFW_SAMPLES, 4);

    // GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    // const GLFWvidmode* vmode = glfwGetVideoMode(monitor);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* vmode = glfwGetVideoMode(monitor);

    window = glfwCreateWindow(g_gl_width, g_gl_height, "SPH - Oculus", NULL, NULL);
    glfwSetWindowPos(window, vmode->width, vmode->height);

    if (!window) {
        fprintf(stderr, "ERROR: could not open window with GLFW3\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, KeyCallback);

    // start GLEW extension handler
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK)
    {
        printf("Error: OpenGL Get Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    if ((hmd = ovrHmd_Create(0)) == NULL)
    {
        fprintf(stderr, "Error: failed to open Oculus HMD, falling back to virtual debug HMD\n");
        if(!(hmd = ovrHmd_CreateDebug(ovrHmd_DK2))) {
            fprintf(stderr, "Error: failed to create virtual debug HMD\n");
            return -1;
        }
        else
        {
            fprintf(stdout, "Log: using virtual debug HMD\n");
            fprintf(stdout, "initialized HMD: %s - %s\n", hmd->Manufacturer, hmd->ProductName);
            debug = true;
        }
    }

    glfwSetWindowSize(window, hmd->Resolution.w, hmd->Resolution.h);
    
    g_gl_width = hmd->Resolution.w;
    g_gl_height = hmd->Resolution.h;

    /* enable position and rotation tracking */
    ovrHmd_ConfigureTracking(hmd, ovrTrackingCap_Orientation | ovrTrackingCap_MagYawCorrection | ovrTrackingCap_Position, 0);

    /* retrieve the optimal render target resolution for each eye */
    eyeres[0] = ovrHmd_GetFovTextureSize(hmd, ovrEye_Left, hmd->DefaultEyeFov[0], 1.0);
    eyeres[1] = ovrHmd_GetFovTextureSize(hmd, ovrEye_Right, hmd->DefaultEyeFov[1], 1.0);

    /* and create a single render target texture to encompass both eyes */
    fb_width = eyeres[0].w + eyeres[1].w;
    fb_height = eyeres[0].h > eyeres[1].h ? eyeres[0].h : eyeres[1].h;
    update_rtarg(fb_width, fb_height);

    /* fill in the ovrGLTexture structures that describe our render target texture */
    for(i=0; i<2; i++) {
        fb_ovr_tex[i].OGL.Header.API = ovrRenderAPI_OpenGL;
        fb_ovr_tex[i].OGL.Header.TextureSize.w = fb_tex_width;
        fb_ovr_tex[i].OGL.Header.TextureSize.h = fb_tex_height;
        /* this next field is the only one that differs between the two eyes */
        fb_ovr_tex[i].OGL.Header.RenderViewport.Pos.x = i == 0 ? 0 : fb_width / 2.0;
        fb_ovr_tex[i].OGL.Header.RenderViewport.Pos.y = 0;
        fb_ovr_tex[i].OGL.Header.RenderViewport.Size.w = fb_width / 2.0;
        fb_ovr_tex[i].OGL.Header.RenderViewport.Size.h = fb_height;
        fb_ovr_tex[i].OGL.TexId = fb_tex;   /* both eyes will use the same texture id */
    }

    /* fill in the ovrGLConfig structure needed by the SDK to draw our stereo pair
     * to the actual HMD display (SDK-distortion mode)
     */
    memset(&glcfg, 0, sizeof(glcfg));
    glcfg.OGL.Header.API = ovrRenderAPI_OpenGL;
    glcfg.OGL.Header.RTSize = hmd->Resolution;
    glcfg.OGL.Header.Multisample = 1;

#ifdef WIN32
    glcfg.OGL.Window = GetActiveWindow();
    glcfg.OGL.DC = wglGetCurrentDC();
#elif defined(__linux__)
    glcfg.OGL.Win = glfwGetX11Window(window);
    glcfg.OGL.Disp = glfwGetX11Display();
#endif

    if(hmd->HmdCaps & ovrHmdCap_ExtendDesktop)
    {
        printf("running in \"extended desktop\" mode\n");
    }
    else
    {
        /* to sucessfully draw to the HMD display in "direct-hmd" mode, we have to
         * call ovrHmd_AttachToWindow
         * XXX: this doesn't work properly yet due to bugs in the oculus 0.4.1 sdk/driver
         */
#ifdef WIN32
        ovrHmd_AttachToWindow(hmd, glcfg.OGL.Window, 0, 0);
#elif defined(__linux__)
        ovrHmd_AttachToWindow(hmd, (void*)glcfg.OGL.Win, 0, 0);
#endif
        printf("running in \"direct-hmd\" mode\n");
    }

    /* enable low-persistence display and dynamic prediction for lattency compensation */
    hmd_caps = ovrHmdCap_LowPersistence | ovrHmdCap_DynamicPrediction;
    ovrHmd_SetEnabledCaps(hmd, hmd_caps);

    /* configure SDK-rendering and enable chromatic abberation correction, vignetting, and
     * timewrap, which shifts the image before drawing to counter any lattency between the call
     * to ovrHmd_GetEyePose and ovrHmd_EndFrame.
     */
    distort_caps = ovrDistortionCap_Chromatic | ovrDistortionCap_Vignette | ovrDistortionCap_TimeWarp | ovrDistortionCap_Overdrive;

    if(!ovrHmd_ConfigureRendering(hmd, &glcfg.Config, distort_caps, hmd->DefaultEyeFov, eye_rdesc)) {
        fprintf(stderr, "failed to configure distortion renderer\n");
    }

    /* disable the retarded "health and safety warning" */
    // ovrhmd_EnableHSWDisplaySDKRender(hmd, 0);

    fprintf(stdout, "%s\n", SEPARATOR);

    /* some rendering defaults */
    glEnable (GL_DEPTH_TEST); // enable depth-testing
    glDepthFunc (GL_LESS); // depth-testing interprets a smaller value as "closer"
    glEnable (GL_CULL_FACE); // cull face
    glCullFace (GL_BACK); // cull back face
    glFrontFace (GL_CCW); // GL_CCW for counter clock-wise
    
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // glClearColor(.8, .8, .8, 1.0);
    InitParticleState(initOption); // initialize particle positions, update position buffer
    
    // initialize gridCellIndex with grid cell point positions, 
    // i.e., each float4, the first 3 floating points are set to the (x, y, z) of the gird cell point
    InitGridCellState();
    
    // glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    glClearColor(.8f, .8f, .8f, 1.0f);

    // glClearDepth(1.0f);

    ////////////////////////////////////////////////////////////////////////////////
    const char* vertex_shader_tri = readShader("shaders/vs_tri.glsl");
    printf("\ntri vertex shader: \n%s\n", vertex_shader_tri);

    unsigned int vs_tri = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs_tri, 1, &vertex_shader_tri, NULL);
    glCompileShader(vs_tri);
    print_shader_info_log(vs_tri);
    
    const char* fragment_shader_tri = readShader("shaders/fs_tri.glsl");
    printf("\ntri fragment shader: \n%s\n", fragment_shader_tri);

    unsigned int fs_tri = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs_tri, 1, &fragment_shader_tri, NULL);
    glCompileShader(fs_tri);
    print_shader_info_log(fs_tri);

    hellotri_program = glCreateProgram();
    glAttachShader(hellotri_program, fs_tri);
    glAttachShader(hellotri_program, vs_tri);
    glLinkProgram(hellotri_program);
    print_shader_program_info_log(hellotri_program);
    
    tri_view_mat_location = glGetUniformLocation(hellotri_program, "view");
    tri_proj_mat_location = glGetUniformLocation(hellotri_program, "proj");
    tri_location = glGetAttribLocation(hellotri_program, "tri");

    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    GLuint positionBuffer;
    glGenBuffers(1, &positionBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(tri_location, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(tri_location);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    const char* vertex_shader = readShader("shaders/vs_sph_billboard.glsl");
    // printf("\nvertex shader: \n%s\n", vertex_shader);

    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, NULL);
    glCompileShader(vs);
    print_shader_info_log(vs);
    
    const char* fragment_shader = readShader("shaders/fs_sph_billboard.glsl");
    // printf("\nfragment shader: \n%s\n", fragment_shader);
  
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader, NULL);
    glCompileShader(fs);
    print_shader_info_log(fs);
    
    shader_program = glCreateProgram();
    glAttachShader(shader_program, fs);
    glAttachShader(shader_program, vs);
    glLinkProgram(shader_program);
    print_shader_program_info_log(shader_program);

    view_mat_location = glGetUniformLocation(shader_program, "view");
    proj_mat_location = glGetUniformLocation(shader_program, "proj");

    glGenBuffers(1, &position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferData(GL_ARRAY_BUFFER, kParticleCount * POS_ATTRIB_SIZE * sizeof(float), position, GL_DYNAMIC_DRAW);
    glGenVertexArrays (1, &vao);
    glBindVertexArray (vao);
    glBindBuffer (GL_ARRAY_BUFFER, position_vbo);
    glVertexAttribPointer (0, 4, GL_FLOAT, GL_FALSE, 0, NULL); 
    glEnableVertexAttribArray(0);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    const char* vertex_shader_triangle = readShader("shaders/vs_mc.glsl");
    // printf("\nvertex shader: \n%s\n", vertex_shader_triangle);;

    unsigned int vs_triMC = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs_triMC, 1, &vertex_shader_triangle, NULL);
    glCompileShader(vs_triMC);
    print_shader_info_log(vs_triMC);

    const char* fragment_shader_triangle = readShader("shaders/fs_mc.glsl");
    // printf("\nfragment shader: \n%s\n", fragment_shader_triangle);

    unsigned int fs_triMC = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs_triMC, 1, &fragment_shader_triangle, NULL);
    glCompileShader(fs_triMC);
    print_shader_info_log(fs_triMC);
    
    triangle_program = glCreateProgram();
    glAttachShader(triangle_program, vs_triMC);
    glAttachShader(triangle_program, fs_triMC);
    
    const char* tess_ctrl_shader = readShader("shaders/tc_ts.glsl");
    // printf("\ntessellation control shader: \n%s\n", tess_ctrl_shader);

    const char* tess_eval_shader = readShader("shaders/te_ts.glsl");
    // printf("\ntessellation evaluation shader: \n%s\n", tess_eval_shader);

    unsigned int tcs = glCreateShader(GL_TESS_CONTROL_SHADER);
    glShaderSource(tcs, 1, &tess_ctrl_shader, NULL);
    glCompileShader(tcs);
    print_shader_info_log(tcs);
    
    unsigned int tes = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(tes, 1, &tess_eval_shader, NULL);
    glCompileShader(tes);
    print_shader_info_log(tes);
    
    glAttachShader(triangle_program, tes);
    glAttachShader(triangle_program, tcs);
    glLinkProgram(triangle_program);
    print_shader_program_info_log(triangle_program);

    outer_tess_fac_loc = glGetUniformLocation(triangle_program, "tess_fac_outer");
    inner_tess_fac_loc = glGetUniformLocation(triangle_program, "tess_fac_inner");

    mc_view_mat_location = glGetUniformLocation(triangle_program, "view");
    mc_proj_mat_location = glGetUniformLocation(triangle_program, "proj");

    // create geometry buffers here
    glGenBuffers(1, &triangle_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, triangle_vbo);
    glBufferData(GL_ARRAY_BUFFER, maxTri * sizeof(float) * 9, tri, GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &norm_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo);
    glBufferData(GL_ARRAY_BUFFER, maxTri * sizeof(float) * 9, norm, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &vaoTri);
    glBindVertexArray(vaoTri);
    glBindBuffer(GL_ARRAY_BUFFER, triangle_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, norm_vbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, NULL);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    return 0;
}

int run()
{
    int err;
    
    err = run_sph_kernel_applyBodyForce();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_applyBodyForce(). Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }
    
    err = run_sph_kernel_advance();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_advance(). Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }
    
    // err = run_sph_kernel_hashparticles();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_sph_kernel_hashparticles. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // err = run_sph_kernel_sort();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_sph_kernel_sort. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // err = run_sph_kernel_sortPostPass();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_sph_kernel_sortPostPass. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // err = run_copy();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_copy. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // err = run_sph_kernel_indexx();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_sph_kernel_indexx. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // err = run_sph_kernel_indexPostPass();
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failure: run_sph_kernel_indexPostPass. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    err = run_sph_kernel_findNeighbors();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_findNeighbors. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_sph_kernel_pressure();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_pressure. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_sph_kernel_calcRelaxPos();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_calcRelaxPos. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_sph_kernel_moveToRelaxPos();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_moveToRelaxPos. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_sph_kernel_resolveCollisions();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_sph_kernel_resolveCollisions. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_mc_kernel_reset();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_mc_kernel_reset. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_mc_kernel_gridval();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_mc_kernel_gridval. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_mc_kernel_gridGrad();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_mc_kernel_gridGrad. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_mc_kernel_cubeindex();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_mc_kernel_cubeindex. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = run_mc_kernel_interpolate();
    if (err != CL_SUCCESS)
    {
        printf("Failure: run_mc_kernel_interpolate. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    clFinish(ComputeCommands);

    if (err != CL_SUCCESS)
        shutdown_opencl();

    return CL_SUCCESS;
}

int setup_compute_devices(int gpu)
{
    int err;
    size_t returned_size;
   
    ComputeDeviceType = gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

#if (USE_GL_ATTACHMENTS)
    printf(SEPARATOR);
    printf("Using active OpenGL context ...\n");

    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

    cl_context_properties properties[] = {
       CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
       (cl_context_properties)kCGLShareGroup, 0
    };
   
    // Create a context from a CGL share group
    //
    ComputeContext = clCreateContext(properties, 0, 0, clLogMessagesToStdoutAPPLE, 0, 0);

    if (!ComputeContext)
    {
        return -2;
    }

#else

    // Connect to a compute device
    //
    err = clGetDeviceIDs(NULL, ComputeDeviceType, 1, &ComputeDeviceId, NULL);
   
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate compute device! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Create a compute context
    //
    ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, clLogMessagesToStdoutAPPLE, NULL, &err);
    if (!ComputeContext || err != CL_SUCCESS)
    {
        printf("Error: Failed to create a compute context.\n");
        return EXIT_FAILURE;
    }
#endif

    size_t device_count;

    cl_device_id device_ids[16];

    err = clGetContextInfo(ComputeContext, CL_CONTEXT_DEVICES, sizeof(device_ids), device_ids, &returned_size);
    if(err)
    {
        printf("Error: Failed to retrieve compute devices for context! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    device_count = returned_size / sizeof(cl_device_id);
    
    int i = 0;
    int device_found = 0;
    cl_device_type device_type; 
    for(i = 0; i < device_count; i++) 
    {
        clGetDeviceInfo(device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
        if(device_type == ComputeDeviceType) 
        {
            ComputeDeviceId = device_ids[i];
            device_found = 1;
            break;
        }   
    }

    if(!device_found)
    {
        printf("Error: Failed to locate compute device!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    //
    ComputeCommands = clCreateCommandQueue(ComputeContext, ComputeDeviceId, 0, &err);
    if (!ComputeCommands)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }

    // Report the device vendor and device name
    //
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    err = clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(ComputeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve device info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    printf(SEPARATOR);
    printf("Connecting to %s %s...\n", vendor_name, device_name);

    return CL_SUCCESS;

}

int setup_compute_memory()
{
    int err;

    printf(SEPARATOR);
    printf("Allocating buffers on compute device\n");

    SPHVelbufferSize = sizeof(float) * kParticleCount * VEL_ATTRIB_SIZE;

    VelocityBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHVelbufferSize, NULL, &err);
    if (!VelocityBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create VelocityBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // SortedVelocityBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHVelbufferSize, NULL, &err);
    // if (!SortedVelocityBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create SortedVelocityBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    SPHPosbufferSize = sizeof(float) * kParticleCount * POS_ATTRIB_SIZE;

    prevPositionBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPosbufferSize, NULL, &err);
    if (!prevPositionBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create prevPositionBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // SortedPrevPositionBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPosbufferSize, NULL, &err);
    // if (!SortedPrevPositionBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create SortedPrevPositionBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // SortedPositionBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPosbufferSize, NULL, &err);
    // if (!SortedPositionBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create SortedPositionBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // SPHIdxbufferSize = sizeof(int) * kParticleCount * PARTICLEIndex_ATTRIB_SIZE;
    // particleIndex = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHIdxbufferSize, NULL, &err);
    // if (!particleIndex || err != CL_SUCCESS)
    // {
    //     printf("Failed to create particleIndex. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // SPHGIdxbufferSize = sizeof(int) * voxel * GRIDCELLIndex_ATTRIB_SIZE;
    // gridCellIndexBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHGIdxbufferSize, NULL, &err);
    // if (!gridCellIndexBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create gridCellIndexBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // gridCellIndexFixedUpBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHGIdxbufferSize, NULL, &err);
    // if (!gridCellIndexFixedUpBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create gridCellIndexFixedUpBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    SPHNeighborbufferSize = sizeof(int) * voxel * NEIGHBOR_MAP_ATTRIB_SIZE;
    neighborMapBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHNeighborbufferSize, NULL, &err);
    if (!neighborMapBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create neighborMapBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    SPHDensitybufferSize = sizeof(float) * kParticleCount * DENSITY_ATTRIB_SIZE;
    densityBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHDensitybufferSize, NULL, &err);
    if (!densityBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create densityBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    nearDensityBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHDensitybufferSize, NULL, &err);
    if (!nearDensityBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create nearDensityBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }    

    SPHPressurebufferSize = sizeof(float) * kParticleCount * PRESSURE_ATTRIB_SIZE;
    pressureBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPressurebufferSize, NULL, &err);
    if (!pressureBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create pressureBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    nearPressureBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPressurebufferSize, NULL, &err);
    if (!nearPressureBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create nearPressureBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    RelaxedPosBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPosbufferSize, NULL, &err);
    if (!RelaxedPosBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create RelaxedPosBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCGridCellbufferSize = sizeof(float) * GRIDCELL_ATTRIB_SIZE * (volNx + 1) * (volNy + 1) * (volNz + 1);
    MCGridBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCGridCellbufferSize, NULL, &err);
    if (!MCGridBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCGridBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCGridGradbufferSize = sizeof(float) * GRIDCELL_GRAD_ATTRIB_SIZE * (volNx + 1) * (volNy + 1) * (volNz + 1);
    MCGridGradBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCGridGradbufferSize, NULL, &err);
    if (!MCGridGradBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCGridGradBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCEdgeTableBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_ONLY, sizeof(int)*256, NULL, &err);
    if (!MCEdgeTableBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCEdgeTableBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCTriTableBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_ONLY, sizeof(int)*256*16, NULL, &err);
    if (!MCTriTableBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCTriTableBuffer. Error %s\n", GetErrorString(err)); 
        return EXIT_FAILURE;
    }

    MCCubeIndexBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, sizeof(int)*volNx*volNy*volNz, NULL, &err);
    if (!MCCubeIndexBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCCubeIndexBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCGridCellIndexbufferSize = sizeof(int) * GRIDPOINT_ATTRIB_SIZE * (volNx) * (volNy) * (volNz);
    MCCubePointIndexBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCGridCellIndexbufferSize, NULL, &err);
    if (!MCCubePointIndexBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCCubePointIndexBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCTriBufferSize = sizeof(TRIANGLE) * maxTri;
    MCTriBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCTriBufferSize, NULL, &err);
    if (!MCTriBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCTriBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCNormBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCTriBufferSize, NULL, &err);
    if (!MCNormBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCNormBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    MCTriCountBufferSize = sizeof(int) * volNx * volNy * volNz;
    MCTriCountBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCTriCountBufferSize, NULL, &err);
    if (!MCTriCountBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create MCTriCountBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

#if (USE_GL_ATTACHMENTS)

    PositionBuffer = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, position_vbo, &err);
    if (!PositionBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create PositionBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // MCTriBuffer = clCreateFromGLBuffer(ComputeContext, CL_MEM_READ_WRITE, triangle_vbo, &err);
    // if (!MCTriBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create MCTriBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

#else

    PositionBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, SPHPosbufferSize, NULL, &err);
    if (!PositionBuffer || err != CL_SUCCESS)
    {
        printf("Failed to create PositionBuffer. Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // MCTriBuffer = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, MCTriBufferSize, NULL, &err);
    // if (!MCTriBuffer || err != CL_SUCCESS)
    // {
    //     printf("Failed to create MCTriBuffer. Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

#endif

    err = clEnqueueWriteBuffer(ComputeCommands, VelocityBuffer, CL_TRUE, 0, SPHVelbufferSize, velocity, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to VelocityBuffer %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(ComputeCommands, MCGridBuffer, CL_TRUE, 0, MCGridCellbufferSize, gridCellIndex, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to MCGridBuffer %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(ComputeCommands, MCEdgeTableBuffer, CL_TRUE, 0, sizeof(int)*256, edgeTable, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to MCEdgeTableBuffer %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(ComputeCommands, MCTriTableBuffer, CL_TRUE, 0, sizeof(int)*256*16, triTable, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to MCTriTableBuffer %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueWriteBuffer(ComputeCommands, MCCubePointIndexBuffer, CL_TRUE, 0, MCGridCellIndexbufferSize, gridPointIndex, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to MCCubePointIndexBuffer %s\n",  GetErrorString(err));
        return EXIT_FAILURE;
    }

    return CL_SUCCESS;

}

int setup_compute_kernels(void)
{
    int err = 0;

    char *source = 0;
    size_t length = 0;

    printf(SEPARATOR);
    printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME);

    err = file_to_string(COMPUTE_KERNEL_FILENAME, &source, &length);
    if (err)
    {
        return -8;
    }

    ComputeProgram = clCreateProgramWithSource(ComputeContext, 1, (const char**) &source, NULL, &err);
    if (!ComputeProgram || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute program. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // Build the program executable
    // 
    printf(SEPARATOR);
    printf("Building compute program...\n");
    err = clBuildProgram(ComputeProgram, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf(SEPARATOR);
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf(SEPARATOR);
        printf("%s\n", buffer);
        printf(SEPARATOR);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from within the program
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_applyBodyForce);  //sph_kernel_applyBodyForce  
    ComputeKernel_sph_kernel_applyBodyForce = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_applyBodyForce, &err);
    if (!ComputeKernel_sph_kernel_applyBodyForce || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_sph_kernel_advance
    // 
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_advance); //sph_kernel_advance
    ComputeKernel_sph_kernel_advance = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_advance, &err);
    if (!ComputeKernel_sph_kernel_advance || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // // ComputeKernel_sph_kernel_hashparticles
    // //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_hashparticles); // sph_kernel_hashparticles
    // ComputeKernel_sph_kernel_hashparticles = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_hashparticles, &err);
    // if (!ComputeKernel_sph_kernel_hashparticles || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // // ComputeKernel_sph_kernel_sort
    // //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_sort); // sph_kernel_sort
    // ComputeKernel_sph_kernel_sort = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_sort, &err);
    // if (!ComputeKernel_sph_kernel_sort || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // // ComputeKernel_sph_kernel_sortPostPass
    // //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_sortPostPass); // sph_kernel_sortPostPass
    // ComputeKernel_sph_kernel_sortPostPass = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_sortPostPass, &err);
    // if (!ComputeKernel_sph_kernel_sortPostPass || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // // copy
    // //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_COPY); // copy
    // ComputeKernel_copy = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_COPY, &err);
    // if (!ComputeKernel_copy || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // // ComputeKernel_sph_kernel_indexx
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_indexx);
    // ComputeKernel_sph_kernel_indexx = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_indexx, &err);
    // if (!ComputeKernel_sph_kernel_indexx || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // // ComputeKernel_sph_kernel_indexPostPass
    // //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_indexPostPass);
    // ComputeKernel_sph_kernel_indexPostPass = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_indexPostPass, &err);
    // if (!ComputeKernel_sph_kernel_indexPostPass || err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // ComputeKernel_sph_kernel_findNeighbors
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_findNeighbors);
    ComputeKernel_sph_kernel_findNeighbors = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_findNeighbors, &err);
    if (!ComputeKernel_sph_kernel_findNeighbors || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_sph_kernel_pressure
    //
    // printf("Creating kernel '%s'..\n", COMPUTE_KERNEL_METHOD_NAME_SPH_pressure);
    ComputeKernel_sph_kernel_pressure = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_pressure, &err);
    if (!ComputeKernel_sph_kernel_pressure || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_sph_kernel_calcRelaxPos
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_calcRelaxPos);
    ComputeKernel_sph_kernel_calcRelaxPos = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_calcRelaxPos, &err);
    if (!ComputeKernel_sph_kernel_calcRelaxPos || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_sph_kernel_moveToRelaxPos
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_moveToRelaxPos);
    ComputeKernel_sph_kernel_moveToRelaxPos = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_SPH_moveToRelaxPos, &err);
    if (!ComputeKernel_sph_kernel_moveToRelaxPos || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_sph_kernel_resolveCollisions
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_resolveCollisions);
    ComputeKernel_sph_kernel_resolveCollisions = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_resolveCollisions, &err);
    if (!ComputeKernel_sph_kernel_resolveCollisions || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_mc_kernel_reset
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_MC_RESET);
    ComputeKernel_mc_kernel_reset = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_MC_RESET, &err);
    if (!ComputeKernel_mc_kernel_reset || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_mc_kernel_gridval
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_MC_gridval);
    ComputeKernel_mc_kernel_gridval = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_MC_gridval, &err);
    if (!ComputeKernel_mc_kernel_gridval || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_mc_kernel_gridGrad
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_MC_gridGrad);
    ComputeKernel_mc_kernel_gridGrad = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_MC_gridGrad, &err);
    if (!ComputeKernel_mc_kernel_gridGrad || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_mc_kernel_cubeindex
    //
    // printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_MC_CUBEINDEX);
    ComputeKernel_mc_kernel_cubeindex = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_MC_CUBEINDEX, &err);
    if (!ComputeKernel_mc_kernel_cubeindex || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // ComputeKernel_VertexInterp
    //
    printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME_MC_INTERPOLATE);
    ComputeKernel_mc_kernel_vertInterp = clCreateKernel(ComputeProgram, COMPUTE_KERNEL_METHOD_NAME_MC_INTERPOLATE, &err);
    if (!ComputeKernel_mc_kernel_vertInterp || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    return CL_SUCCESS;
}

// sph_kernel_applyBodyForce
int run_sph_kernel_applyBodyForce()
{
    int err = 0;

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;
    err  = clSetKernelArg(ComputeKernel_sph_kernel_applyBodyForce, 0, sizeof(int), &kParticleCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_applyBodyForce, 1, sizeof(cl_mem), &VelocityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_applyBodyForce, 2, sizeof(float), &kDt);
    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: sph_kernel_applyBodyForce. Error %s\n", GetErrorString(err));
        exit(1);
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_applyBodyForce, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_applyBodyForce, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_applyBodyForce, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_applyBodyForce. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, VelocityBuffer, CL_TRUE, 0, SPHVelbufferSize, velocity, 0, NULL, NULL);

    // printf(SEPARATOR);
    // printf("After kernel:\n");
    // printVelocity(velocity);

    return 0;
}

int run_sph_kernel_advance()
{
    int err = 0;
    
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;
    err  = clSetKernelArg(ComputeKernel_sph_kernel_advance, 0, sizeof(int), &kParticleCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_advance, 1, sizeof(float), &kDt);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_advance, 2, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_advance, 3, sizeof(cl_mem), &prevPositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_advance, 4, sizeof(cl_mem), &VelocityBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: sph_kernel_advance. Error %s\n", GetErrorString(err));
        exit(1);
    }

#if (USE_GL_ATTACHMENTS)

    err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
        return -1;
    }

#endif

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_advance, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_advance, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_advance, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_applyBodyForce. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

//     // printf(SEPARATOR);
//     // printf("After kernel:\n");
//     // printPosition(position);
//     // printf(SEPARATOR);

    return 0;
}

// static int run_sph_kernel_hashparticles()
// {
//     int err = 0;

//     size_t global;                      // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 0, sizeof(float), &kViewWidth);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 1, sizeof(float), &kViewHeight);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 2, sizeof(float), &kViewDepth);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 3, sizeof(int), &Nx);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 4, sizeof(int), &Ny);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 5, sizeof(int), &Nz);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 6, sizeof(float), &kCellSize);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 7, sizeof(cl_mem), &PositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 8, sizeof(cl_mem), &particleIndex);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_hashparticles, 9, sizeof(cl_mem), &VelocityBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: sph_kernel_hashparticles. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

// #if (USE_GL_ATTACHMENTS)

//     err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
//         return -1;
//     }

// #endif

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_hashparticles, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_hashparticles, (int)global, (int)local);

//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_hashparticles, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_hashparticles. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     // err = clEnqueueReadBuffer(ComputeCommands, particleIndex, CL_TRUE, 0, SPHIdxbufferSize, partIdx, 0, NULL, NULL);

//     // printf(SEPARATOR);
//     // printf("Print Part-index\n");
//     // int i;
//     // for (i = 0; i < kParticleCount; i++)
//     // {
//     //     printf("Particle %d -- (%d, %d)\n", i, partIdx[2*i+0], partIdx[2*i+1]);
//     // }

//     return 0;
// }

// static int run_sph_kernel_sort()
// {
//     int err = 0;

//     size_t global;                     // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;
    
//     err  = clSetKernelArg(ComputeKernel_sph_kernel_sort, 0, sizeof(cl_mem), &particleIndex);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sort, 3, sizeof(int), &sortOrder);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: sph_kernel_sort. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

//     int stage;
//     int passOfStage;

//     err = 0;
//     for (stage = 0; stage < numStages; stage++)
//     {
//         err = clSetKernelArg(ComputeKernel_sph_kernel_sort, 1, sizeof(int), &stage);

//         for (passOfStage = 0; passOfStage < stage + 1; passOfStage++)
//         {
//             err |= clSetKernelArg(ComputeKernel_sph_kernel_sort, 2, sizeof(int), &passOfStage);

//             // Get the maximum work group size for executing the kernel on the device
//             //
//             err |= clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_sort, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//             if (err != CL_SUCCESS)
//             {
//                 printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//                 return EXIT_FAILURE;
//             }

//             global = kParticleCount/2;
//             global = local > global ? local : global;

//             // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_sort, (int)global, (int)local);

//             err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_sort, 1, NULL, &global, &local, 0, NULL, NULL);
//             if (err)
//             {
//                 printf("Failed to execute kernel sph_kernel_sort. Error: %s\n", GetErrorString(err));
//                 return EXIT_FAILURE;
//             }
//         }
//     }

//     // err = clEnqueueReadBuffer(ComputeCommands, particleIndex, CL_TRUE, 0, SPHIdxbufferSize, partIdx, 0, NULL, NULL);

//     // printf(SEPARATOR);
//     // printf("Print Part-index\n");
//     // int i;
//     // for (i = 0; i < kParticleCount; i++)
//     // {
//     //     printf("Particle %d -- (%d, %d)\n", i, partIdx[2*i+0], partIdx[2*i+1]);
//     // }

//     return 0;
// }

// static int run_sph_kernel_sortPostPass()
// {
//     int err = 0;

//     size_t global;                     // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 0, sizeof(cl_mem), &PositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 1, sizeof(cl_mem), &VelocityBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 2, sizeof(cl_mem), &particleIndex);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 3, sizeof(cl_mem), &SortedPositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 4, sizeof(cl_mem), &SortedVelocityBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 5, sizeof(cl_mem), &prevPositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_sortPostPass, 6, sizeof(cl_mem), &SortedPrevPositionBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: sph_kernel_sortPostPass. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

// // #if (USE_GL_ATTACHMENTS)

// //     err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
// //     if (err != CL_SUCCESS)
// //     {
// //         printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
// //         return -1;
// //     }

// // #endif

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_sortPostPass, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_sortPostPass, (int)global, (int)local);

//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_sortPostPass, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_sortPostPass. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     return 0;
// }

// static int run_copy()
// {
//     int err = 0;

//     size_t global;                     // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_copy, 0, sizeof(cl_mem), &PositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_copy, 1, sizeof(cl_mem), &SortedPositionBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: kernel_copy. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_copy, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_COPY, (int)global, (int)local);

//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_copy, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_sortPostPass. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_copy, 0, sizeof(cl_mem), &VelocityBuffer);
//     err |= clSetKernelArg(ComputeKernel_copy, 1, sizeof(cl_mem), &SortedVelocityBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: kernel_copy. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_copy, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_COPY, (int)global, (int)local);
    
//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_copy, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_sortPostPass. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     return 0;
// }

// static int run_sph_kernel_indexx()
// {
//     int err = 0;

//     size_t global;                     // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_sph_kernel_indexx, 0, sizeof(float), &kParticleCount);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_indexx, 1, sizeof(cl_mem), &PositionBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_indexx, 2, sizeof(cl_mem), &gridCellIndexBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_indexx. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_indexx, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_indexx, (int)global, (int)local);

//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_indexx, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_indexx. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     return 0;
// }

// static int run_sph_kernel_indexPostPass()
// {
//     int err = 0;

//     size_t global;                     // global domain size for our calculation
//     size_t local;                       // local domain size for our calculation

//     err = 0;

//     err  = clSetKernelArg(ComputeKernel_sph_kernel_indexPostPass, 0, sizeof(int), &kGridCellCount);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_indexPostPass, 1, sizeof(cl_mem), &gridCellIndexBuffer);
//     err |= clSetKernelArg(ComputeKernel_sph_kernel_indexPostPass, 2, sizeof(cl_mem), &gridCellIndexFixedUpBuffer);

//     if (err != CL_SUCCESS)
//     {
//         printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_indexPostPass. Error %s\n", GetErrorString(err));
//         exit(1);
//     }

//     // Get the maximum work group size for executing the kernel on the device
//     //
//     err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_indexPostPass, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
//     if (err != CL_SUCCESS)
//     {
//         printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     global = kParticleCount;
//     global = local > global ? local : global;

//     // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_indexPostPass, (int)global, (int)local);

//     err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_indexPostPass, 1, NULL, &global, &local, 0, NULL, NULL);
//     if (err)
//     {
//         printf("Failed to execute kernel sph_kernel_indexPostPass. Error: %s\n", GetErrorString(err));
//         return EXIT_FAILURE;
//     }

//     return 0;
// }

int run_sph_kernel_findNeighbors()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 0, sizeof(int), &kParticleCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 1, sizeof(float), &kViewWidth);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 2, sizeof(float), &kViewHeight);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 3, sizeof(float), &kViewDepth);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 4, sizeof(int), &Nx);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 5, sizeof(int), &Ny);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 6, sizeof(int), &Nz);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 7, sizeof(float), &kCellSize);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 8, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 9, sizeof(cl_mem), &neighborMapBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 10, sizeof(int), &kMaxNeighbourCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_findNeighbors, 11, sizeof(float), &kEpsilon);
    
    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_findNeighbors. Error %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_findNeighbors, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_findNeighbors, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_findNeighbors, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_findNeighbors. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    return 0;
}

static int run_sph_kernel_pressure()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;
    
    err  = clSetKernelArg(ComputeKernel_sph_kernel_pressure, 0, sizeof(float), &mass);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 1, sizeof(float), &kCellSize);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 2, sizeof(float), &kNorm);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 3, sizeof(float), &kNearNorm);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 4, sizeof(int), &kMaxNeighbourCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 5, sizeof(float), &kStiffness);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 6, sizeof(float), &kNearStiffness);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 7, sizeof(float), &kRestDensity);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 8, sizeof(float), &kEpsilon);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 9, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 10, sizeof(cl_mem), &neighborMapBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 11, sizeof(cl_mem), &densityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 12, sizeof(cl_mem), &nearDensityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 13, sizeof(cl_mem), &pressureBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_pressure, 14, sizeof(cl_mem), &nearPressureBuffer);
    
    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_findNeighbors. Error %s\n", GetErrorString(err));
        exit(1);
    }
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_pressure, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_pressure, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_pressure, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_pressure. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, densityBuffer, CL_TRUE, 0, SPHDensitybufferSize, density, 0, NULL, NULL);

    // printf(SEPARATOR);
    // printf("Print density\n");
    // int i;
    // for (i = 0; i < kParticleCount; i++)
    // {
    //     printf("particle %d local density: %f\n", i, density[i]);
    // }

    return 0;
}

int run_sph_kernel_calcRelaxPos()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 0, sizeof(float), &mass);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 1, sizeof(float), &kCellSize);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 2, sizeof(float), &kDt);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 3, sizeof(float), &kDt2);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 4, sizeof(float), &kNearNorm);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 5, sizeof(float), &kNorm);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 6, sizeof(float), &kSurfaceTension);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 7, sizeof(float), &kLinearViscocity);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 8, sizeof(float), &kQuadraticViscocity);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 9, sizeof(int), &kMaxNeighbourCount);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 10, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 11, sizeof(cl_mem), &VelocityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 12, sizeof(cl_mem), &neighborMapBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 13, sizeof(cl_mem), &densityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 14, sizeof(cl_mem), &nearDensityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 15, sizeof(cl_mem), &pressureBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 16, sizeof(cl_mem), &nearPressureBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_calcRelaxPos, 17, sizeof(cl_mem), &RelaxedPosBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_findNeighbors. Error %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_calcRelaxPos, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_calcRelaxPos, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_calcRelaxPos, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_calcRelaxPos. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    return 0;
}

int run_sph_kernel_moveToRelaxPos()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_sph_kernel_moveToRelaxPos, 0, sizeof(float), &kDt);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_moveToRelaxPos, 1, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_moveToRelaxPos, 2, sizeof(cl_mem), &prevPositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_moveToRelaxPos, 3, sizeof(cl_mem), &VelocityBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_moveToRelaxPos, 4, sizeof(cl_mem), &RelaxedPosBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_moveToRelaxPos. Error %s\n", GetErrorString(err));
        exit(1);
    }

#if (USE_GL_ATTACHMENTS)

    err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
        return -1;
    }

#endif

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_moveToRelaxPos, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_SPH_moveToRelaxPos, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_moveToRelaxPos, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_moveToRelaxPos. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    return 0;

}

int run_sph_kernel_resolveCollisions()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 0, sizeof(float), &kDt);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 1, sizeof(float), &kParticleRadius);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 2, sizeof(float), &kCellSize);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 3, sizeof(float), &kViewWidth);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 4, sizeof(float), &kViewHeight);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 5, sizeof(float), &kViewDepth);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 6, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_sph_kernel_resolveCollisions, 7, sizeof(cl_mem), &VelocityBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_sph_kernel_resolveCollisions. Error %s\n", GetErrorString(err));
        exit(1);
    }

#if (USE_GL_ATTACHMENTS)

    err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
        return -1;
    }

#endif

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_sph_kernel_resolveCollisions, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;
    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_resolveCollisions, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_sph_kernel_resolveCollisions, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel sph_kernel_resolveCollisions. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, PositionBuffer, CL_TRUE, 0, SPHPosbufferSize, position, 0, NULL, NULL);

    return 0;
}

int run_mc_kernel_reset()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_mc_kernel_reset, 0, sizeof(int), &volInd);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_reset, 1, sizeof(cl_mem), &MCGridBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_mc_kernel_reset. Error %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_mc_kernel_reset, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = volInd;

    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_MC_RESET, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_mc_kernel_reset, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel mc_kernel_reset. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    // err = clEnqueueReadBuffer(ComputeCommands, MCGridBuffer, CL_TRUE, 0, MCGridCellbufferSize, gridCellIndex, 0, NULL, NULL);

    return 0;
}

int run_mc_kernel_gridval()
{
    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;
    
    err  = clSetKernelArg(ComputeKernel_mc_kernel_gridval, 0, sizeof(float), &mass);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 1, sizeof(float), &kCellSize);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 2, sizeof(float), &kNorm);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 3, sizeof(float), &kNearNorm);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 4, sizeof(float), &kEpsilon);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 5, sizeof(cl_mem), &PositionBuffer);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 6, sizeof(cl_mem), &MCGridBuffer);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridval, 7, sizeof(int), &volInd);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_mc_kernel_gridval. Error %s\n", GetErrorString(err));
        exit(1);
    }

#if (USE_GL_ATTACHMENTS)

    err = clEnqueueAcquireGLObjects(ComputeCommands, 1, &PositionBuffer, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to attach Position Buffer. Error %s\n", GetErrorString(err));
        return -1;
    }

#endif

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_mc_kernel_gridval, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = kParticleCount;

    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_MC_gridval, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_mc_kernel_gridval, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel mc_kernel_gridval. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, MCGridBuffer, CL_TRUE, 0, MCGridCellbufferSize, gridCellIndex, 0, NULL, NULL);

    return 0;
}

int run_mc_kernel_gridGrad()
{
    // int i, j, k;

    // for (k = 1; k < volNz; k++)
    // {
    //     for (j = 1; j < volNy; j++)
    //     {
    //         for (i = 1; i < volNx; i++)
    //         {
    //             gridCellGrad[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+0] = 
    //                 (gridCellIndex[4*arrayIndexFromCoordinate(i-1, volNx+1, j, volNy+1, k)+3] // -1
    //                     - gridCellIndex[4*arrayIndexFromCoordinate(i+1, volNx+1, j, volNy+1, k)+3])/volEdgeX; //+1

    //             gridCellGrad[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+1] = 
    //                 (gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j-1, volNy+1, k)+3] // -(volNy+1)
    //                     - gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j+1, volNy+1, k)+3])/volEdgeY; // +(volNy+1)

    //             gridCellGrad[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+2] = 
    //                 (gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k-1)+3] // -(volNx+1)*(volNy+1)
    //                     - gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k+1)+3])/volEdgeZ; // +(volNx+1)*(volNy+1)
    //         }
    //     }
    // }

    int err = 0;

    size_t global;                     // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 0, sizeof(int), &volNx);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 1, sizeof(int), &volNy);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 2, sizeof(int), &volNz);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 3, sizeof(float), &volEdgeX);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 4, sizeof(float), &volEdgeY);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 5, sizeof(float), &volEdgeZ);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 6, sizeof(cl_mem), &MCGridBuffer);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_gridGrad, 7, sizeof(cl_mem), &MCGridGradBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_mc_kernel_gridGrad. Error %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_mc_kernel_gridGrad, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }
    
    global = nvNx * nvNy * nvNz; // (volNx + 1) * (volNy + 1) * (volNz +1)

    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_MC_gridGrad, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_mc_kernel_gridGrad, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel mc_kernel_gridGrad. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, MCGridGradBuffer, CL_TRUE, 0, MCGridGradbufferSize, gridCellGrad, 0, NULL, NULL);

    return 0;
}

static XYZ VertexInterp(float isolevel,XYZ p1,XYZ p2,float valp1,float valp2)
{
   float mu;
   XYZ p;

   if (ABS(isolevel-valp1) < 0.00001)
      return(p1);
   if (ABS(isolevel-valp2) < 0.00001)
      return(p2);
   if (ABS(valp1-valp2) < 0.00001)
      return(p1);
   mu = (isolevel - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}

static XYZ GradInterp(float isolevel, XYZ g1, XYZ g2, float valp1, float valp2)
{
    float mu;
    XYZ g;

    if (ABS(isolevel-valp1) < 0.00001)
        return (g1);
    if (ABS(isolevel-valp2) < 0.00001)
        return (g2);
    if (ABS(valp1-valp2) < 0.00001)
        return (g1);

    mu = (isolevel - valp1) / (valp2 - valp1);
    g.x = g1.x + mu * (g2.x - g1.x);
    g.y = g1.y + mu * (g2.y - g1.y);
    g.z = g1.z + mu * (g2.z - g1.z);

    return(g);
}

int run_mc_kernel_cubeindex()
{
    // int i;
    // for (i = 0; i < (volNx) * (volNy) * (volNz); i++)
    // {
    //     int idx0 = gridPointIndex[8*i+0];
    //     int idx1 = gridPointIndex[8*i+1];
    //     int idx2 = gridPointIndex[8*i+2];
    //     int idx3 = gridPointIndex[8*i+3];
    //     int idx4 = gridPointIndex[8*i+4];
    //     int idx5 = gridPointIndex[8*i+5];
    //     int idx6 = gridPointIndex[8*i+6];
    //     int idx7 = gridPointIndex[8*i+7];

    //     float val0 = gridCellIndex[4*idx4+3];
    //     float val1 = gridCellIndex[4*idx5+3];
    //     float val2 = gridCellIndex[4*idx1+3];
    //     float val3 = gridCellIndex[4*idx0+3];
    //     float val4 = gridCellIndex[4*idx7+3];
    //     float val5 = gridCellIndex[4*idx6+3];
    //     float val6 = gridCellIndex[4*idx2+3];
    //     float val7 = gridCellIndex[4*idx3+3];
        
    //     int cidx = 0;
    //     if (val0 < isoval) cidx |= 1;
    //     if (val1 < isoval) cidx |= 2;
    //     if (val2 < isoval) cidx |= 4;
    //     if (val3 < isoval) cidx |= 8;
    //     if (val4 < isoval) cidx |= 16;
    //     if (val5 < isoval) cidx |= 32;
    //     if (val6 < isoval) cidx |= 64;
    //     if (val7 < isoval) cidx |= 128;

    //     cubeIndex[i] = cidx;
    // }

    ////////////////////////////////////////////////////////////////////////////////////////////////    
    int err = 0;

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err = 0;

    err  = clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 0, sizeof(float), &isoval);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 1, sizeof(int), &volNx);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 2, sizeof(int), &volNy);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 3, sizeof(int), &volNz);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 4, sizeof(cl_mem), &MCGridBuffer);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 5, sizeof(cl_mem), &MCCubePointIndexBuffer);
    err |= clSetKernelArg(ComputeKernel_mc_kernel_cubeindex, 6, sizeof(cl_mem), &MCCubeIndexBuffer);

    if (err != CL_SUCCESS)
    {
        printf("Failed to set kernel arguments: ComputeKernel_mc_kernel_cubeindex. Error %s\n", GetErrorString(err));
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(ComputeKernel_mc_kernel_cubeindex, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    global = nvNx * nvNy * nvNz; // (volNx + 1) * (volNy + 1) * (volNz +1)

    global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_MC_CUBEINDEX, (int)global, (int)local);

    err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_mc_kernel_cubeindex, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Failed to execute kernel mc_kernel_cubeindex. Error: %s\n", GetErrorString(err));
        return EXIT_FAILURE;
    }

    err = clEnqueueReadBuffer(ComputeCommands, MCCubeIndexBuffer, CL_TRUE, 0, sizeof(int)*volNx*volNy*volNz, cubeIndex, 0, NULL, NULL);
    
    return 0;
}

int run_mc_kernel_interpolate()
{
    XYZ vertlist[12];
    XYZ normlist[12];
    
    Ntri = 0;
    
    memset(tri, 0.0, sizeof(TRIANGLE) * maxTri);
    memset(norm, 0.0, sizeof(TRIANGLE) * maxTri);
    memset(normFlat, 0.0, sizeof(TRIANGLE) * maxTri);

    // memset(triFull, 0.0, sizeof(TRIANGLE) * maxTri);
    // memset(normFull, 0.0, sizeof(TRIANGLE) * maxTri);
    // memset(triFullCopy, 0.0, sizeof(TRIANGLE) * maxTri);
    // memset(normFullCopy, 0.0, sizeof(TRIANGLE) * maxTri);
    // memset(triCount, 0, sizeof(int)*volNx*volNy*volNz);

    int i;
    for (i = 0; i < (volNx) * (volNy) * (volNz); i++)
    {   
        int idx0 = gridPointIndex[8*i+0];
        int idx1 = gridPointIndex[8*i+1];
        int idx2 = gridPointIndex[8*i+2];
        int idx3 = gridPointIndex[8*i+3];
        int idx4 = gridPointIndex[8*i+4];
        int idx5 = gridPointIndex[8*i+5];
        int idx6 = gridPointIndex[8*i+6];
        int idx7 = gridPointIndex[8*i+7];

        float val0 = gridCellIndex[4*idx4+3];
        float val1 = gridCellIndex[4*idx5+3];
        float val2 = gridCellIndex[4*idx1+3];
        float val3 = gridCellIndex[4*idx0+3];
        float val4 = gridCellIndex[4*idx7+3];
        float val5 = gridCellIndex[4*idx6+3];
        float val6 = gridCellIndex[4*idx2+3];
        float val7 = gridCellIndex[4*idx3+3];
        
        XYZ P0, P1, P2, P3, P4, P5, P6, P7;
        P0.x = gridCellIndex[4*idx4+0]; P0.y = gridCellIndex[4*idx4+1]; P0.z = gridCellIndex[4*idx4+2];
        P1.x = gridCellIndex[4*idx5+0]; P1.y = gridCellIndex[4*idx5+1]; P1.z = gridCellIndex[4*idx5+2];
        P2.x = gridCellIndex[4*idx1+0]; P2.y = gridCellIndex[4*idx1+1]; P2.z = gridCellIndex[4*idx1+2];
        P3.x = gridCellIndex[4*idx0+0]; P3.y = gridCellIndex[4*idx0+1]; P3.z = gridCellIndex[4*idx0+2];
        P4.x = gridCellIndex[4*idx7+0]; P4.y = gridCellIndex[4*idx7+1]; P4.z = gridCellIndex[4*idx7+2];
        P5.x = gridCellIndex[4*idx6+0]; P5.y = gridCellIndex[4*idx6+1]; P5.z = gridCellIndex[4*idx6+2];
        P6.x = gridCellIndex[4*idx2+0]; P6.y = gridCellIndex[4*idx2+1]; P6.z = gridCellIndex[4*idx2+2];
        P7.x = gridCellIndex[4*idx3+0]; P7.y = gridCellIndex[4*idx3+1]; P7.z = gridCellIndex[4*idx3+2];

        XYZ G0, G1, G2, G3, G4, G5, G6, G7;
        G0.x = gridCellGrad[4*idx4+0]; G0.y = gridCellGrad[4*idx4+1]; G0.z = gridCellGrad[4*idx4+2];
        G1.x = gridCellGrad[4*idx5+0]; G1.y = gridCellGrad[4*idx5+1]; G1.z = gridCellGrad[4*idx5+2];
        G2.x = gridCellGrad[4*idx1+0]; G2.y = gridCellGrad[4*idx1+1]; G2.z = gridCellGrad[4*idx1+2];
        G3.x = gridCellGrad[4*idx0+0]; G3.y = gridCellGrad[4*idx0+1]; G3.z = gridCellGrad[4*idx0+2];
        G4.x = gridCellGrad[4*idx7+0]; G4.y = gridCellGrad[4*idx7+1]; G4.z = gridCellGrad[4*idx7+2];
        G5.x = gridCellGrad[4*idx6+0]; G5.y = gridCellGrad[4*idx6+1]; G5.z = gridCellGrad[4*idx6+2];
        G6.x = gridCellGrad[4*idx2+0]; G6.y = gridCellGrad[4*idx2+1]; G6.z = gridCellGrad[4*idx2+2];
        G7.x = gridCellGrad[4*idx3+0]; G7.y = gridCellGrad[4*idx3+1]; G7.z = gridCellGrad[4*idx3+2];

        int edgetable = edgeTable[cubeIndex[i]];
        if (edgetable == 0)
        {
            continue;
        }

        if (edgetable & 1)
        {
            vertlist[0] = VertexInterp(isoval, P0, P1, val0, val1);
            normlist[0] = GradInterp(isoval, G0, G1, val0, val1);
        }

        if (edgetable & 2)
        {
            vertlist[1] = VertexInterp(isoval, P1, P2, val1, val2);
            normlist[1] = GradInterp(isoval, G1, G2, val1, val2);
        }

        if (edgetable & 4)
        {
            vertlist[2] = VertexInterp(isoval, P2, P3, val2, val3);
            normlist[2] = GradInterp(isoval, G2, G3, val2, val3);
        }

        if (edgetable & 8)
        {
            vertlist[3] = VertexInterp(isoval, P3, P0, val3, val0);
            normlist[3] = GradInterp(isoval, G3, G0, val3, val0);
        }
        
        if (edgetable & 16)
        {
            vertlist[4] = VertexInterp(isoval, P4, P5, val4, val5);
            normlist[4] = GradInterp(isoval, G4, G5, val4, val5);
        }
        
        if (edgetable & 32)
        {
            vertlist[5] = VertexInterp(isoval, P5, P6, val5, val6);
            normlist[5] = GradInterp(isoval, G5, G6, val5, val6);
        }

        if (edgetable & 64)
        {
            vertlist[6] = VertexInterp(isoval, P6, P7, val6, val7);
            normlist[6] = GradInterp(isoval, G6, G7, val6, val7);
        }

        if (edgetable & 128)
        {
            vertlist[7] = VertexInterp(isoval, P7, P4, val7, val4);
            normlist[7] = GradInterp(isoval, G7, G4, val7, val4);
        }
        
        if (edgetable & 256)
        {
            vertlist[8] = VertexInterp(isoval, P0, P4, val0, val4);
            normlist[8] = GradInterp(isoval, G0, G4, val0, val4);
        }

        if (edgetable & 512)
        {
            vertlist[9] = VertexInterp(isoval, P1, P5, val1, val5);
            normlist[9] = GradInterp(isoval, G1, G5, val1, val5);
        }

        if (edgetable & 1024)
        {
            vertlist[10] = VertexInterp(isoval, P2, P6, val2, val6);
            normlist[10] = GradInterp(isoval, G2, G6, val2, val6);
        }

        if (edgetable & 2048)
        {
            vertlist[11] = VertexInterp(isoval, P3, P7, val3, val7);
            normlist[11] = GradInterp(isoval, G3, G7, val3, val7);
        }

        int j;
        int ntri = 0;
        int l;

        TRIANGLE triangles[10];
        TRIANGLE normals[10];
        TRIANGLE normalsFlat[10];

        for (j = 0; triTable[cubeIndex[i]][j] != -1; j+=3)
        {
            triangles[ntri].p[0] = vertlist[triTable[cubeIndex[i]][j+0]];
            triangles[ntri].p[1] = vertlist[triTable[cubeIndex[i]][j+1]];
            triangles[ntri].p[2] = vertlist[triTable[cubeIndex[i]][j+2]];

            //////////////////////////////////////////////////////////////////////////////////////////////    

            XYZ P01, P02;
            P01.x = triangles[ntri].p[1].x - triangles[ntri].p[0].x;
            P01.y = triangles[ntri].p[1].y - triangles[ntri].p[0].y;
            P01.z = triangles[ntri].p[1].z - triangles[ntri].p[0].z;

            P02.x = triangles[ntri].p[2].x - triangles[ntri].p[0].x;
            P02.y = triangles[ntri].p[2].y - triangles[ntri].p[0].y;
            P02.z = triangles[ntri].p[2].z - triangles[ntri].p[0].z;

            // P02 cross P01
            XYZ NORM;
            NORM.x = -P02.y * P01.z + P02.z * P01.y;
            NORM.y = -P02.z * P01.x + P02.x * P01.z;
            NORM.z = -P02.x * P01.y + P02.y * P01.x;
            normalsFlat[ntri].p[0] = NORM;
            normalsFlat[ntri].p[1] = NORM;
            normalsFlat[ntri].p[2] = NORM;

            //////////////////////////////////////////////////////////////////////////////////////////////

            normals[ntri].p[0] = normlist[triTable[cubeIndex[i]][j+0]];
            normals[ntri].p[1] = normlist[triTable[cubeIndex[i]][j+1]];
            normals[ntri].p[2] = normlist[triTable[cubeIndex[i]][j+2]];

            ntri++;
        }

        for (l=0; l<ntri; l++)
        {
            tri[Ntri+l] = triangles[l];
            norm[Ntri+l] = normals[l];
            normFlat[Ntri+l] = normalsFlat[l];

            triIdx[3*Ntri+3*l+0] = 3*Ntri+3*l+0;
            triIdx[3*Ntri+3*l+1] = 3*Ntri+3*l+1;
            triIdx[3*Ntri+3*l+2] = 3*Ntri+3*l+2;
        }
        
        Ntri += ntri;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////    
    // int err = 0;

    // size_t global;                      // global domain size for our calculation
    // size_t local;                       // local domain size for our calculation

    // err = 0;
    
    // err  = clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 0, sizeof(float), &isoval);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 1, sizeof(int), &volNx);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 2, sizeof(int), &volNy);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 3, sizeof(int), &volNz);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 4, sizeof(cl_mem), &MCCubePointIndexBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 5, sizeof(cl_mem), &MCGridBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 6, sizeof(cl_mem), &MCGridGradBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 7, sizeof(cl_mem), &MCCubeIndexBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 8, sizeof(cl_mem), &MCEdgeTableBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 9, sizeof(cl_mem), &MCTriTableBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 10, sizeof(cl_mem), &MCTriBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 11, sizeof(cl_mem), &MCTriCountBuffer);
    // err |= clSetKernelArg(ComputeKernel_mc_kernel_vertInterp, 12, sizeof(cl_mem), &MCNormBuffer);
    
    // if (err != CL_SUCCESS)
    // {
    //     printf("Failed to set kernel arguments: ComputeKernel_mc_kernel_vertInterp. Error %s\n", GetErrorString(err));
    //     exit(1);
    // }

    // // Get the maximum work group size for executing the kernel on the device
    // //
    // err = clGetKernelWorkGroupInfo(ComputeKernel_mc_kernel_vertInterp, ComputeDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    // if (err != CL_SUCCESS)
    // {
    //     printf("Error: Failed to retrieve kernel work group info! Error %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }

    // global = nvNx * nvNy * nvNz; // (volNx + 1) * (volNy + 1) * (volNz +1)

    // global = local > global ? local : global;

    // printf("Executing '%s' (global='%d, local='%d')...\n", COMPUTE_KERNEL_METHOD_NAME_MC_INTERPOLATE, (int)global, (int)local);

    // err = clEnqueueNDRangeKernel(ComputeCommands, ComputeKernel_mc_kernel_vertInterp, 1, NULL, &global, &local, 0, NULL, NULL);
    // if (err)
    // {
    //     printf("Failed to execute kernel mc_kernel_interpolate. Error: %s\n", GetErrorString(err));
    //     return EXIT_FAILURE;
    // }
    
    // // err = clEnqueueReadBuffer(ComputeCommands, MCTriBuffer, CL_TRUE, 0, maxTri*sizeof(TRIANGLE), triFull, 0, NULL, NULL);
    // err = clEnqueueReadBuffer(ComputeCommands, MCNormBuffer, CL_TRUE, 0, maxTri*sizeof(TRIANGLE), normFull, 0, NULL, NULL);

    // err = clEnqueueReadBuffer(ComputeCommands, MCTriCountBuffer, CL_TRUE, 0, sizeof(int)*volNx*volNy*volNz, triCount, 0, NULL, NULL);

    // // int p;
    // // for (p = 0; p < maxTri; p++)
    // // {
    // //     printf(SEPARATOR);
    // //     printf("triFull - [(%f, %f, %f) - (%f, %f, %f) - (%f, %f, %f)]\n", triFull[p].p[0].x, triFull[p].p[0].y, triFull[p].p[0].z, 
    // //                                                            triFull[p].p[1].x, triFull[p].p[1].y, triFull[p].p[1].z,
    // //                                                            triFull[p].p[2].x, triFull[p].p[2].y, triFull[p].p[2].z);
    // //     printf(SEPARATOR);
    // // }
    
    // printf(SEPARATOR);
    // printf("Packing:...\n");
    // int pack;
    // for (pack = 0; pack < volNx*volNy*volNz; pack++)
    // {
    //     int sum, triTotal = 0;
    //     for (sum = 0; sum < pack; sum++)
    //     {
    //         triTotal += triCount[sum];
    //     }

    //     int n = triCount[pack];

    //     printf("voxel %d - #Tri %d - starts at %d\n", pack, n, triTotal);

    //     int j;
    //     for (j = 0; j < n; j++)
    //     {
    //         // triFullCopy[triTotal+j] = triFull[5*pack+j];
    //         // normFullCopy[triTotal+j] = normFull[5*pack+j];
            
    //         norm[triTotal+j] = normFull[5*pack+j];
    //     }
    // }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////    

    return 0;
}

int setup_opencl(int use_gpu)
{
   printf(SEPARATOR);
   printf("Setting up Compute...\n");

   int err;
   err = setup_compute_devices(use_gpu);
   
   if (err != CL_SUCCESS)
   {
       return err;
   }
   
   err = setup_compute_memory();
   if (err != CL_SUCCESS)
   {
       return err;
   }
   
   err = setup_compute_kernels();
   if (err != CL_SUCCESS)
   {
       return err;
   }
   
   int temp = 0;
   for (temp = kParticleCount; temp > 1; temp >>= 1)
        ++numStages;
   
   return CL_SUCCESS;
}

void InitSPHParticle(int idx, float px, float py, float pz, float vx, float vy, float vz)
{
    int ii = 4*idx;

    position[ii + 0] = px;
    position[ii + 1] = py;
    position[ii + 2] = pz;
    position[ii + 3] = 0;

    velocity[ii + 0] = vx;
    velocity[ii + 1] = vy;
    velocity[ii + 2] = vz;
    velocity[ii + 3] = 0.0;
}

void InitBreakDam()
{
   int count = 0;

    float i, j, k;

    for (i = -kViewWidth/2+kParticleRadius+kCellSize; i < kViewWidth/2-kParticleRadius-kCellSize; i+=kCellSize/2)
    {
        for (j = -kViewHeight/2+kParticleRadius+kCellSize; j < kViewHeight/4-kParticleRadius-kCellSize; j+=kCellSize/2)
        {
            for (k = -kViewDepth/2+kParticleRadius+kCellSize; k < kViewDepth/4-kParticleRadius-kCellSize; k+=kCellSize/2)
            {
                if (count < kParticleCount)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    // printf("position: particle %d - (%f, %f, %f, %f)\n", count, position[4*count+0], position[4*count+1], position[4*count+2], position[4*count+3]);
                    
                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    // printf("velocity: particle %d - (%f, %f, %f, %f)\n", count, velocity[4*count+0], velocity[4*count+1], velocity[4*count+2], velocity[4*count+3]);

                    count++;
                }
            }
        }
    }

    printf("%d particles initialized\n", count);
}

void InitMidAirDrop()
{
    int count = 0;

    float i, j, k;
    
    for (j = -kViewHeight/2+kParticleRadius+kCellSize; j < kViewHeight/2-kParticleRadius-kCellSize; j+=kCellSize/2)
    {
        for (i = -kViewWidth/4+kParticleRadius+kCellSize; i < kViewWidth/4-kParticleRadius-kCellSize; i+=kCellSize/2)
        {
            for (k = -kViewDepth/4+kParticleRadius+kCellSize; k < kViewDepth/4-kParticleRadius-kCellSize; k+=kCellSize/2)
            {
                if (count < kParticleCount)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    // printf("position: particle %d - (%f, %f, %f, %f)\n", count, position[4*count+0], position[4*count+1], position[4*count+2], position[4*count+3]);
                    
                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    // printf("velocity: particle %d - (%f, %f, %f, %f)\n", count, velocity[4*count+0], velocity[4*count+1], velocity[4*count+2], velocity[4*count+3]);

                    count++;
                }
            }
        }
    }
}

void InitTwoCubes()
{
    int count = 0;

    float i, j, k;
    for (j = -kViewHeight/2+kParticleRadius+kCellSize; j < kViewHeight/2-kParticleRadius-kCellSize; j+=kCellSize/2)
    {
        for (i = 0+kParticleRadius+kCellSize; i < kViewWidth/2-kParticleRadius-kCellSize; i+=kCellSize/2)    
        {
            for (k = -kViewDepth/2+kParticleRadius+kCellSize; k < 0-kParticleRadius-kCellSize; k+=kCellSize/2)
            {
                if (count < kParticleCount/3)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    // printf("position: particle %d - (%f, %f, %f, %f)\n", count, position[4*count+0], position[4*count+1], position[4*count+2], position[4*count+3]);
                    
                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    // printf("velocity: particle %d - (%f, %f, %f, %f)\n", count, velocity[4*count+0], velocity[4*count+1], velocity[4*count+2], velocity[4*count+3]);

                    count++;
                }
            }
        }
    }

    for (j = -kViewHeight/2+kParticleRadius+kCellSize; j < kViewHeight/2-kParticleRadius-kCellSize; j+=kCellSize/2)
    {
        for (i = -kViewWidth/2+kParticleRadius+kCellSize; i < 0-kParticleRadius-kCellSize; i+=kCellSize/2)
        {
            for (k = 0+kParticleRadius+kCellSize; k < kViewDepth/2-kParticleRadius-kCellSize; k+=kCellSize/2)
            {
                if (count < kParticleCount)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    count++;
                }
            }
        }
    }
}

void InitBallDrop()
{
    int count = 0;

    float i, j, k;

    for (k = -kViewDepth/4+kParticleRadius+kCellSize; k < kViewDepth/2-kParticleRadius-kCellSize; k+=kCellSize/2)
    {
        for (i = -kViewWidth/4+kParticleRadius+kCellSize; i < kViewWidth/4-kParticleRadius-kCellSize; i+=kCellSize/2)
        {
            for (j = kViewHeight/4+kParticleRadius+kCellSize; j < kViewHeight/2-kParticleRadius-kCellSize; j+=kCellSize/2)    
            {
                if (count < kParticleCount/12)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    count++;
                }
            }
        }
    }
    
    for (j = -kViewHeight/2+kParticleRadius+kCellSize; j < 0-kParticleRadius-kCellSize; j+=kCellSize/2)
    {
        for (k = -kViewDepth/2+kParticleRadius+kCellSize; k < kViewDepth/2-kParticleRadius-kCellSize; k+=kCellSize/2)    
        {
            for (i = -kViewWidth/2+kParticleRadius+kCellSize; i < kViewWidth/2-kParticleRadius-kCellSize; i+=kCellSize/2)                
            {
                if (count < kParticleCount)
                {
                    position[4*count+0] = i;
                    position[4*count+1] = j;
                    position[4*count+2] = k;
                    position[4*count+3] = 0.0;

                    velocity[4*count+0] = 0.0;
                    velocity[4*count+1] = 0.0;
                    velocity[4*count+2] = 0.0;
                    velocity[4*count+3] = 0.0;

                    count++;
                }
            }
        }
    }
}

void InitParticleState(int initOption)
{
    // InitBreakDam(); // 1000 frames
    // InitMidAirDrop(); // 1000 frames
    // InitTwoCubes(); // 800 frames - do 1000 anyways
    // InitBallDrop(); // 900 frames - dito
    
    switch(initOption) {
        
        case 1:
            InitBreakDam(); 
            break;
        case 2:
            InitMidAirDrop(); 
            break;
        case 3:
            InitTwoCubes(); 
            break;
        case 4:
            InitBallDrop();
            break;
        default:
            InitBreakDam(); 
            break;
    }
}

void InitGridCellState()
{
    int i, j, k;
    for (i = 0; i < volNx + 1; i++)
    {
        for (j = 0; j < volNy + 1; j++)
        {
            for (k = 0; k < volNz + 1; k++)
            {
                gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+0] = -kViewWidth/2  + volEdgeX * i;
                gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+1] = -kViewHeight/2 + volEdgeY * j;
                gridCellIndex[4*arrayIndexFromCoordinate(i, volNx+1, j, volNy+1, k)+2] = -kViewDepth/2  + volEdgeX * k;   
            }
        }
    }

    int count = 0;
    
    for (i = 0; i < volNx; i++)
    {
        for (j = 0; j < volNy; j++)
        {
            for (k = 0; k < volNz; k++)    
            {
                /*
                       7--------6     *---4----*
                      /|       /|    /|       /|
                     / |      / |   7 |      5 |
                    /  |     /  |  /  8     /  9
                   3--------2   | *----6---*   |
                   |   |    |   | |   |    |   |
                   |   4----|---5 |   *---0|---*
                   |  /     |  /  11 /     10 /
                   | /      | /   | 3      | 1
                   |/       |/    |/       |/
                   0--------1     *---2----*
                */
                gridPointIndex[8*count+0] = arrayIndexFromCoordinate(i,   volNx+1, j,   volNy+1, k);
                gridPointIndex[8*count+1] = arrayIndexFromCoordinate(i+1, volNx+1, j,   volNy+1, k);
                gridPointIndex[8*count+2] = arrayIndexFromCoordinate(i+1, volNx+1, j+1, volNy+1, k);
                gridPointIndex[8*count+3] = arrayIndexFromCoordinate(i,   volNx+1, j+1, volNy+1, k);
                gridPointIndex[8*count+4] = arrayIndexFromCoordinate(i,   volNx+1, j,   volNy+1, k+1);
                gridPointIndex[8*count+5] = arrayIndexFromCoordinate(i+1, volNx+1, j,   volNy+1, k+1);
                gridPointIndex[8*count+6] = arrayIndexFromCoordinate(i+1, volNx+1, j+1, volNy+1, k+1);
                gridPointIndex[8*count+7] = arrayIndexFromCoordinate(i,   volNx+1, j+1, volNy+1, k+1);

                count++;
            }
        }
    }
}

void cleanup(void)
{
    if(hmd) {
        ovrHmd_Destroy(hmd);
    }
    ovr_Shutdown();
}

void reshape(int width, int height) {
    g_gl_width = width;
    g_gl_height = height;
}

void update_rtarg(int width, int height)
{
    if (!fbo)
    {
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &fb_tex);
        glGenRenderbuffers(1, &fb_depth);

        glBindTexture(GL_TEXTURE_2D, fb_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    fb_tex_width = next_pow2(width);
    fb_tex_height = next_pow2(height);

    /* create and attach the texture that will be used as a color buffer */
    glBindTexture(GL_TEXTURE_2D, fb_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fb_tex_width, fb_tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb_tex, 0);

    /* create and attach the renderbuffer that will serve as our z-buffer */
    glBindRenderbuffer(GL_RENDERBUFFER, fb_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, fb_tex_width, fb_tex_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fb_depth);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Error: incomplete framebuffer!\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    fprintf(stdout, "Log: created render target: %dx%d (texture size: %dx%d)\n", width, height, fb_tex_width, fb_tex_height);

}

unsigned int next_pow2(unsigned int x)
{
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

static void KeyCallback(GLFWwindow* p_Window, int p_Key, int p_Scancode, int p_Action, int p_Mods)
{
    if (p_Key == GLFW_KEY_ESCAPE && p_Action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(p_Window, GL_TRUE);
    } else
    {
        ovrhmd_EnableHSWDisplaySDKRender(hmd, 0);
        if (p_Key == GLFW_KEY_A && p_Action == GLFW_PRESS)
        {
            cam_pos[0] -= cam_speed * elapsedseconds;
            cam_moved = true;
        }

        if (p_Key == GLFW_KEY_D && p_Action == GLFW_PRESS)
        {
            cam_pos[0] += cam_speed * elapsedseconds;
            cam_moved = true;
        }

        if (p_Key == GLFW_KEY_W && p_Action == GLFW_PRESS)
        {
            cam_pos[2] -= cam_speed * elapsedseconds;
            cam_moved = true;
        }

        if (p_Key == GLFW_KEY_S && p_Action == GLFW_PRESS)
        {
            cam_pos[2] += cam_speed * elapsedseconds;
            cam_moved = true;
        }

        if (p_Key == GLFW_KEY_SPACE && p_Action == GLFW_PRESS)
        {
            isrunning = true;
        }

        if (p_Key == GLFW_KEY_ENTER && p_Action == GLFW_PRESS)
        {
            isrunning = false;
        }

        if (p_Key == GLFW_KEY_P && p_Action == GLFW_PRESS)
        {
            renderPart = true;
        }

        if (p_Key == GLFW_KEY_O && p_Action == GLFW_PRESS)
        {
            renderPart = false;
        }

        if (p_Key == GLFW_KEY_E && p_Action == GLFW_PRESS)
        {
            simpleTri = true;
        }

        if (p_Key == GLFW_KEY_R && p_Action == GLFW_PRESS)
        {
            simpleTri = false;
        }

        if (p_Key == GLFW_KEY_1 && p_Action == GLFW_PRESS)
        {
            blinn = true;
        }

        if (p_Key == GLFW_KEY_L && p_Action == GLFW_PRESS) // wireframe
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }

        if (p_Key == GLFW_KEY_K && p_Action == GLFW_PRESS) // faces
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        if (p_Key == GLFW_KEY_M && p_Action == GLFW_PRESS)
        {
            printMesh();
        }

        if (p_Key == GLFW_KEY_N && p_Action == GLFW_PRESS) 
        {
            normalFlat = true;
        }

        if (p_Key == GLFW_KEY_B && p_Action == GLFW_PRESS) 
        {
            normalFlat = false;
        }

        if (p_Key == GLFW_KEY_T && p_Action == GLFW_PRESS) 
        {
            if (!t_was_down) {
                inner_tess_fac += 1.0f;
                printf ("inner tess. factor = %.1f\n", inner_tess_fac);
                t_was_down = true;
                glUniform1f (inner_tess_fac_loc, inner_tess_fac);
              }
        } else 
        {
            t_was_down = false;
        }
        
        if (p_Key == GLFW_KEY_G && p_Action == GLFW_PRESS) 
        {
            if (!g_was_down) {
                inner_tess_fac -= 1.0f;
                printf ("inner tess. factor = %.1f\n", inner_tess_fac);
                g_was_down = true;
                glUniform1f (inner_tess_fac_loc, inner_tess_fac);
              }
        } else 
        {
            g_was_down = false;
        }
        
        if (p_Key == GLFW_KEY_Y && p_Action == GLFW_PRESS) 
        {
            if (!y_was_down) {
                outer_tess_fac += 1.0f;
                printf ("outer tess. factor = %.1f\n", outer_tess_fac);
                y_was_down = true;
                glUniform1f (outer_tess_fac_loc, outer_tess_fac);
              }
        } else 
        {
            y_was_down = false;
        }
        
        if (p_Key == GLFW_KEY_H && p_Action == GLFW_PRESS)
        {
            if (!h_was_down) {
                outer_tess_fac -= 1.0f;
                printf ("outer tess. factor = %.1f\n", outer_tess_fac);
                h_was_down = true;
                glUniform1f (outer_tess_fac_loc, outer_tess_fac);
              }
        } else 
        {
            h_was_down = false;
        }
    }
}