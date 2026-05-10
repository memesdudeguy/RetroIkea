cmake_minimum_required(VERSION 3.20)
project(vulkan_game VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Portability (e.g. Linux aarch64, Apple Silicon, embedded ARM boards)
# - vulkan_game: C++17 + Vulkan + SDL2 — no x86-specific code in src/; Assimp/miniaudio build for target arch.
# - Shaders: need `glslc` on PATH or under VULKAN_SDK (see find_program below).
# - Runtime: prefer GPU drivers with Vulkan 1.1+; bake asset paths or use env for non-x86 install layouts.
# ---------------------------------------------------------------------------

include(FetchContent)
# GLM (header-only): keep builds reproducible on hosts without system glm dev package.
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
  cpphttplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(cpphttplib)

set(RP3D_COMPILE_TESTBED OFF CACHE BOOL "RetroIkea: skip ReactPhysics3D OpenGL testbed" FORCE)
set(RP3D_COMPILE_TESTS OFF CACHE BOOL "RetroIkea: skip ReactPhysics3D unit tests" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/build_rp3d_fetch"
                 "${CMAKE_CURRENT_BINARY_DIR}/reactphysics3d_build" EXCLUDE_FROM_ALL)

FetchContent_Declare(
  vulkan_headers
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG v1.4.341
)
FetchContent_MakeAvailable(vulkan_headers)
# Bundled zlib/minizip: system minizip often lacks unzip.h on the include path Assimp expects.
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_VIEW OFF CACHE BOOL "" FORCE)
set(ASSIMP_IGNORE_GIT_HASH ON CACHE BOOL "" FORCE)
# GCC 15 -Warray-bounds on X3DGeoHelper.cpp (empty vector .back()) trips -Werror; Assimp v5.4.3 is otherwise fine.
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  assimp
  GIT_REPOSITORY https://github.com/assimp/assimp.git
  GIT_TAG v5.4.3
)
# MinGW Windows: default static Assimp (single exe-friendly). Optionally full dynamic/runtime DLLs beside RetroIkea.exe.
option(VULKAN_GAME_WINDOWS_ALL_DYNAMIC
  "MinGW cross-compile: shared libAssimp + dynamic libgcc/libstdc++/winpthread (copy DLLs next to RetroIkea.exe). Use a fresh build dir when toggling."
  OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  if(VULKAN_GAME_WINDOWS_ALL_DYNAMIC)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    message(STATUS "vulkan_game: Windows cross-build — BUILD_SHARED_LIBS=ON (dynamic Assimp DLL)")
  else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    message(STATUS "vulkan_game: Windows cross-build — BUILD_SHARED_LIBS=OFF for static Assimp")
  endif()
endif()
FetchContent_MakeAvailable(assimp)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  set(VULKAN_CROSS_LIBRARY "" CACHE FILEPATH "Path to Windows Vulkan import library (vulkan-1.lib or libvulkan-1.a)")
  set(VULKAN_CROSS_INCLUDE_DIR "" CACHE PATH "Path to Vulkan headers (directory containing vulkan/vulkan.h)")
  set(VULKAN_GAME_MINGW_VULKAN_DLL "" CACHE FILEPATH "Optional path to vulkan-1.dll to bundle next to RetroIkea.exe")
  # MinGW cross builds often need an explicit Windows Vulkan import library path.
  set(_VULKAN_CROSS_HINTS "")
  if(DEFINED ENV{VULKAN_SDK})
    list(APPEND _VULKAN_CROSS_HINTS
      "$ENV{VULKAN_SDK}/Lib"
      "$ENV{VULKAN_SDK}/lib"
      "$ENV{VULKAN_SDK}/Bin"
      "$ENV{VULKAN_SDK}/bin")
  endif()
  list(APPEND _VULKAN_CROSS_HINTS
    "/usr/x86_64-w64-mingw32/lib"
    "/usr/i686-w64-mingw32/lib")
  set(_VULKAN_CROSS_LIB_RESOLVED "${VULKAN_CROSS_LIBRARY}")
  if(NOT _VULKAN_CROSS_LIB_RESOLVED)
    find_library(_VULKAN_CROSS_LIB_RESOLVED NAMES vulkan-1 vulkan HINTS ${_VULKAN_CROSS_HINTS})
  endif()
  if(_VULKAN_CROSS_LIB_RESOLVED)
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
      IMPORTED_LOCATION "${_VULKAN_CROSS_LIB_RESOLVED}")
    message(STATUS "vulkan_game: cross Windows Vulkan lib: ${_VULKAN_CROSS_LIB_RESOLVED}")
    set(_VULKAN_CROSS_INC_RESOLVED "${VULKAN_CROSS_INCLUDE_DIR}")
    if(_VULKAN_CROSS_INC_RESOLVED AND NOT EXISTS "${_VULKAN_CROSS_INC_RESOLVED}/vulkan/vulkan.h")
      message(WARNING "VULKAN_CROSS_INCLUDE_DIR does not contain vulkan/vulkan.h: ${_VULKAN_CROSS_INC_RESOLVED}")
      set(_VULKAN_CROSS_INC_RESOLVED "")
    endif()
    if(NOT _VULKAN_CROSS_INC_RESOLVED)
      if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/Include/vulkan/vulkan.h")
        set(_VULKAN_CROSS_INC_RESOLVED "$ENV{VULKAN_SDK}/Include")
      elseif(EXISTS "/usr/x86_64-w64-mingw32/include/vulkan/vulkan.h")
        set(_VULKAN_CROSS_INC_RESOLVED "/usr/x86_64-w64-mingw32/include")
      elseif(EXISTS "${vulkan_headers_SOURCE_DIR}/include/vulkan/vulkan.h")
        set(_VULKAN_CROSS_INC_RESOLVED "${vulkan_headers_SOURCE_DIR}/include")
      endif()
    endif()
    if(_VULKAN_CROSS_INC_RESOLVED)
      include_directories(SYSTEM "${_VULKAN_CROSS_INC_RESOLVED}")
      message(STATUS "vulkan_game: cross Vulkan include dir: ${_VULKAN_CROSS_INC_RESOLVED}")
    else()
      message(FATAL_ERROR
        "Windows cross-build: Vulkan headers not found.\n"
        "Set VULKAN_CROSS_INCLUDE_DIR to a directory containing vulkan/vulkan.h, "
        "or install Vulkan headers.")
    endif()
  else()
    message(FATAL_ERROR
      "Windows cross-build: Vulkan import library not found.\n"
      "Install a MinGW Vulkan package (libvulkan-1.a) or set VULKAN_SDK so CMake can use "
      "VULKAN_SDK/Lib/vulkan-1.lib.")
  endif()
else()
  find_package(Vulkan REQUIRED)
endif()
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(Threads REQUIRED)

find_program(GLSL_COMPILER glslc
  HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
  DOC "Khronos glslc (Vulkan SDK); required to compile shaders")
if(NOT GLSL_COMPILER)
  message(FATAL_ERROR "glslc not found. Install Vulkan SDK or set GLSL_COMPILER to the glslc path.")
endif()
set(VULKAN_GAME_ASSETS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(VULKAN_GAME_ASSETS_DIR "${VULKAN_GAME_ASSETS_SOURCE_DIR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  # Runtime on installed Windows should resolve assets relative to RetroIkea.exe.
  set(VULKAN_GAME_ASSETS_DIR "assets")
endif()
set(VULKAN_GAME_MESHY_REEXPORT_DIR "${VULKAN_GAME_ASSETS_SOURCE_DIR}/meshy_reexport")
set(SHADER_DIR ${VULKAN_GAME_ASSETS_SOURCE_DIR}/shaders)
set(VULKAN_GAME_SHADER_RUNTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  # Installed Windows layout keeps compiled SPIR-V under {app}\assets\shaders.
  set(VULKAN_GAME_SHADER_RUNTIME_DIR "assets/shaders")
endif()

# Cross builds from Linux otherwise embed /home/.../RetroIkea/... into the .exe; the retail install only has
# portable paths next to RetroIkea.exe. Map any path under CMAKE_CURRENT_SOURCE_DIR to a repo-relative path.
function(vulkan_portable_data_path out_var path_in)
  set(_result "${path_in}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND path_in)
    get_filename_component(_abs "${path_in}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(RELATIVE_PATH _rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_abs}")
    if(NOT _rel MATCHES "^\\.\\.")
      string(REPLACE "\\" "/" _result "${_rel}")
    endif()
  endif()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

set(SHADER_SOURCES ${SHADER_DIR}/shader.vert ${SHADER_DIR}/shader.frag ${SHADER_DIR}/shader_staff.vert
                   ${SHADER_DIR}/post.vert ${SHADER_DIR}/post.frag)

foreach(SHADER ${SHADER_SOURCES})
  get_filename_component(FNAME ${SHADER} NAME)
  set(SPV ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.spv)
  add_custom_command(
    OUTPUT ${SPV}
    COMMAND ${GLSL_COMPILER} ${SHADER} -o ${SPV}
    DEPENDS ${SHADER}
    COMMENT "Compiling ${FNAME}"
  )
  list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders DEPENDS ${SPV_OUTPUTS})

add_executable(vulkan_game
  src/main.cpp
  src/audio.cpp
  src/employee_mesh.cpp
  src/staff_skin.cpp
  src/staff_rp3d_ragdoll.cpp
  src/net_p2p.cpp
  src/lobby_http.cpp
  src/portable_path.cpp
  src/self_update.cpp
)

if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set_target_properties(vulkan_game PROPERTIES OUTPUT_NAME RetroIkea)
endif()

add_dependencies(vulkan_game shaders)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  add_custom_command(TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_game>/assets/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_OUTPUTS} "$<TARGET_FILE_DIR:vulkan_game>/assets/shaders"
    COMMENT "Copy SPIR-V shaders next to RetroIkea.exe for installer/runtime")
endif()

target_include_directories(vulkan_game PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${glm_SOURCE_DIR}
  ${cpphttplib_SOURCE_DIR}
)

find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  target_compile_definitions(vulkan_game PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

# Public lobby origin. Leave empty by default so stale/broken deployments do not show "Lobby HTTP 404";
# set RETRO_IKEA_LOBBY_URL at runtime or configure -DRETRO_IKEA_DEFAULT_LOBBY_URL=https://your-service.
set(RETRO_IKEA_DEFAULT_LOBBY_URL ""
    CACHE STRING "HTTPS lobby origin when RETRO_IKEA_LOBBY_URL env is unset (no trailing slash); empty disables")
if(NOT RETRO_IKEA_DEFAULT_LOBBY_URL STREQUAL "")
  target_compile_definitions(vulkan_game PRIVATE
    "RETRO_IKEA_DEFAULT_LOBBY_URL=\"${RETRO_IKEA_DEFAULT_LOBBY_URL}\"")
endif()
set(RETRO_IKEA_RELEASE_TAG "beta-10"
    CACHE STRING "GitHub release tag baked into the Windows self-updater")
target_compile_definitions(vulkan_game PRIVATE
  "RETRO_IKEA_RELEASE_TAG=\"${RETRO_IKEA_RELEASE_TAG}\"")

# MinGW OpenSSL runtime DLLs (required next to RetroIkea.exe when CPPHTTPLIB_OPENSSL_SUPPORT is on).
set(VULKAN_GAME_MINGW_OPENSSL_SSL_DLL "" CACHE FILEPATH
    "MinGW: path to libssl *.dll to bundle (optional if found under /usr/x86_64-w64-mingw32/bin)")
set(VULKAN_GAME_MINGW_OPENSSL_CRYPTO_DLL "" CACHE FILEPATH
    "MinGW: path to libcrypto *.dll to bundle (optional if found under /usr/x86_64-w64-mingw32/bin)")

# All shipped art/audio under vulkan_game/assets/ (see assets/models, assets/audio, assets/textures).
# Prefer compressed loop when present (see scripts/shrink_audio_assets.sh); else ship WAV.
if(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/store_ambient_loop.mp3")
  set(STORE_AMBIENT_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/store_ambient_loop.mp3")
else()
  set(STORE_AMBIENT_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/store_ambient_loop.wav")
endif()
set(POWER_SWITCH_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/power_switch.mp3")
set(HORROR_AMBIENT_MP3 "${VULKAN_GAME_ASSETS_DIR}/audio/horror_ambient.mp3")
set(CHASE_MUSIC_MP3 "${VULKAN_GAME_ASSETS_DIR}/audio/chase_music.mp3")
set(LIGHT_SWITCH_ON_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/light_switch_on.mp3")
set(STAFF_SPOTTED_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/staff_spotted.mp3")
# Chase / “store is closed, get out” VO: `store_closed_chase_vo.mp3` (e.g. from `ffmpeg -i store-closed.m4a …`).
# Miniaudio’s stock decoders handle MP3/WAV/FLAC, not M4A/AAC directly.
if(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/store_closed_chase_vo.mp3")
  set(STAFF_CHASE_VO_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/store_closed_chase_vo.mp3")
else()
  set(STAFF_CHASE_VO_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/staff_chase_vo.mp3")
endif()

# Staff / player Meshy rig clips (GLB names are short; same skeleton across clips).
set(VULKAN_GAME_STAFF_MODEL_DIR "${VULKAN_GAME_ASSETS_DIR}/models/character")
set(VULKAN_GAME_MESHY_IDLE_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/idle.glb")
set(VULKAN_GAME_MESHY_RUN_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/sprint.glb")
set(VULKAN_GAME_MESHY_WALK_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/walk.glb")
set(VULKAN_GAME_MESHY_SLIDE_RIGHT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/slide_right.glb")
set(VULKAN_GAME_MESHY_CROUCH_WALK_LEFT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/crouch_walk_left.glb")
set(VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_FWD_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/cautious_crouch_forward.glb")
set(VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_BACK_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/cautious_crouch_backward.glb")
set(VULKAN_GAME_MESHY_CROUCH_WALK_RIGHT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/crouch_walk_right.glb")
set(VULKAN_GAME_MESHY_SLIDE_LIGHT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/slide_light.glb")
set(VULKAN_GAME_MESHY_STEP_PUSH_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/step_push.glb")
set(VULKAN_GAME_MESHY_CROUCH_IDLE_BOW_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/crouch_idle_bow.glb")
set(VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/ledge_ladder_climb.glb")
# Optional Meshy slow-ladder clip: prefer bundled reexport; optional user path.
set(VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH "" CACHE FILEPATH "Optional external slow-ladder Meshy GLB")
set(_MESHY_SLOW_LADDER_REEXPORT "${VULKAN_GAME_MESHY_REEXPORT_DIR}/Meshy_AI_biped_Animation_Slow_Ladder_Climb_withSkin.glb")
if(EXISTS "${_MESHY_SLOW_LADDER_REEXPORT}")
  set(VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB "${_MESHY_SLOW_LADDER_REEXPORT}")
elseif(VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH AND EXISTS "${VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH}")
  set(VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB "${VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH}")
endif()
vulkan_portable_data_path(VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB "${VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB}")
if(EXISTS "${_MESHY_SLOW_LADDER_REEXPORT}")
  message(STATUS "vulkan_game: using Meshy slow ladder GLB (meshy_reexport): ${VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB}")
elseif(VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH AND EXISTS "${VULKAN_GAME_MESHY_SLOW_LADDER_EXT_PATH}")
  message(STATUS "vulkan_game: using external Meshy slow ladder GLB: ${VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB}")
endif()
set(VULKAN_GAME_MESHY_LEDGE_GRAB_WALL_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/ledge_grab_wall.glb")
set(VULKAN_GAME_MESHY_SHIMMY_LEFT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/ledge_shimmy_left.glb")
set(VULKAN_GAME_MESHY_SHIMMY_RIGHT_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/ledge_shimmy_right.glb")
set(VULKAN_GAME_JUMPING_FBX "${VULKAN_GAME_STAFF_MODEL_DIR}/jump_regular.glb")
set(VULKAN_GAME_JUMP_RUN_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/jump_run.glb")
set(VULKAN_GAME_MESHY_PUNCH_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/punch_combo.glb")
set(VULKAN_GAME_MESHY_KICK_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/kick.glb")
set(VULKAN_GAME_MESHY_FALL_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/fall_knockdown.glb")
set(VULKAN_GAME_MESHY_STANDUP_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/stand_up.glb")
set(VULKAN_GAME_MESHY_LAND_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/land_hit_reaction.glb")
set(VULKAN_GAME_MESHY_HAIR_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/hair_shove_reaction.glb")
# EXISTS must use VULKAN_GAME_ASSETS_SOURCE_DIR: when cross-compiling, VULKAN_GAME_ASSETS_DIR is the
# portable prefix "assets" only — CMake would resolve EXISTS "assets/..." against the build dir and miss idle.glb.
if(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/models/character/idle.glb")
  set(VULKAN_GAME_EMPLOYEE_FBX "${VULKAN_GAME_ASSETS_DIR}/models/character/idle.glb")
  message(STATUS "vulkan_game staff mesh: ${VULKAN_GAME_ASSETS_DIR}/models/character/idle.glb")
elseif(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/models/character_base.glb")
  set(VULKAN_GAME_EMPLOYEE_FBX "${VULKAN_GAME_ASSETS_DIR}/models/character_base.glb")
  message(STATUS "vulkan_game staff mesh: fallback ${VULKAN_GAME_EMPLOYEE_FBX}")
else()
  set(VULKAN_GAME_EMPLOYEE_FBX "${VULKAN_GAME_ASSETS_DIR}/models/character_base.fbx")
  message(STATUS "vulkan_game staff mesh: fallback ${VULKAN_GAME_EMPLOYEE_FBX}")
endif()

target_compile_definitions(vulkan_game PRIVATE
  "VULKAN_GAME_VERSION_STRING=\"${PROJECT_VERSION}\""
  SHADER_DIR="${VULKAN_GAME_SHADER_RUNTIME_DIR}"
  "VULKAN_GAME_ASSETS_DIR=\"${VULKAN_GAME_ASSETS_DIR}\""
  "VULKAN_GAME_STORE_AMBIENT_WAV=\"${STORE_AMBIENT_WAV}\""
  "VULKAN_GAME_POWER_SWITCH_SFX=\"${POWER_SWITCH_SFX}\""
  "VULKAN_GAME_HORROR_AMBIENT_MP3=\"${HORROR_AMBIENT_MP3}\""
  "VULKAN_GAME_CHASE_MUSIC_MP3=\"${CHASE_MUSIC_MP3}\""
  "VULKAN_GAME_LIGHT_SWITCH_ON_SFX=\"${LIGHT_SWITCH_ON_SFX}\""
  "VULKAN_GAME_STAFF_SPOTTED_SFX=\"${STAFF_SPOTTED_SFX}\""
  "VULKAN_GAME_STAFF_CHASE_VO_SFX=\"${STAFF_CHASE_VO_SFX}\""
  "VULKAN_GAME_EMPLOYEE_FBX=\"${VULKAN_GAME_EMPLOYEE_FBX}\""
  "VULKAN_GAME_MESHY_IDLE_GLB=\"${VULKAN_GAME_MESHY_IDLE_GLB}\""
  "VULKAN_GAME_MESHY_RUN_GLB=\"${VULKAN_GAME_MESHY_RUN_GLB}\""
  "VULKAN_GAME_MESHY_WALK_GLB=\"${VULKAN_GAME_MESHY_WALK_GLB}\""
  "VULKAN_GAME_MESHY_SLIDE_RIGHT_GLB=\"${VULKAN_GAME_MESHY_SLIDE_RIGHT_GLB}\""
  "VULKAN_GAME_MESHY_CROUCH_WALK_LEFT_GLB=\"${VULKAN_GAME_MESHY_CROUCH_WALK_LEFT_GLB}\""
  "VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_FWD_GLB=\"${VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_FWD_GLB}\""
  "VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_BACK_GLB=\"${VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_BACK_GLB}\""
  "VULKAN_GAME_MESHY_CROUCH_WALK_RIGHT_GLB=\"${VULKAN_GAME_MESHY_CROUCH_WALK_RIGHT_GLB}\""
  "VULKAN_GAME_MESHY_SLIDE_LIGHT_GLB=\"${VULKAN_GAME_MESHY_SLIDE_LIGHT_GLB}\""
  "VULKAN_GAME_MESHY_STEP_PUSH_GLB=\"${VULKAN_GAME_MESHY_STEP_PUSH_GLB}\""
  "VULKAN_GAME_MESHY_CROUCH_IDLE_BOW_GLB=\"${VULKAN_GAME_MESHY_CROUCH_IDLE_BOW_GLB}\""
  "VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB=\"${VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB}\""
  "VULKAN_GAME_MESHY_LEDGE_GRAB_WALL_GLB=\"${VULKAN_GAME_MESHY_LEDGE_GRAB_WALL_GLB}\""
  "VULKAN_GAME_MESHY_SHIMMY_LEFT_GLB=\"${VULKAN_GAME_MESHY_SHIMMY_LEFT_GLB}\""
  "VULKAN_GAME_MESHY_SHIMMY_RIGHT_GLB=\"${VULKAN_GAME_MESHY_SHIMMY_RIGHT_GLB}\""
  "VULKAN_GAME_JUMPING_FBX=\"${VULKAN_GAME_JUMPING_FBX}\""
  "VULKAN_GAME_JUMP_RUN_GLB=\"${VULKAN_GAME_JUMP_RUN_GLB}\""
  "VULKAN_GAME_MESHY_PUNCH_GLB=\"${VULKAN_GAME_MESHY_PUNCH_GLB}\""
  "VULKAN_GAME_MESHY_KICK_GLB=\"${VULKAN_GAME_MESHY_KICK_GLB}\""
  "VULKAN_GAME_MESHY_FALL_GLB=\"${VULKAN_GAME_MESHY_FALL_GLB}\""
  "VULKAN_GAME_MESHY_STANDUP_GLB=\"${VULKAN_GAME_MESHY_STANDUP_GLB}\""
  "VULKAN_GAME_MESHY_LAND_GLB=\"${VULKAN_GAME_MESHY_LAND_GLB}\""
  "VULKAN_GAME_STAFF_MELEE_IMPACT_MP3=\"${VULKAN_GAME_ASSETS_DIR}/audio/sfx_staff_melee_impact.mp3\""
  "VULKAN_GAME_PIZZA_SLICE_GLB=\"${VULKAN_GAME_ASSETS_DIR}/models/pizza_slice.glb\""
)

# Title screen loop (prefer bundled assets; fallback to external Downloads only if missing).
set(_VULK_TITLE_MENU_WAV "")
if(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/New_Project.wav")
  set(_VULK_TITLE_MENU_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/New_Project.wav")
elseif(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/the_epic_vista.mp3")
  set(_VULK_TITLE_MENU_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/the_epic_vista.mp3")
elseif(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/the_long_hall.wav")
  set(_VULK_TITLE_MENU_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/the_long_hall.wav")
elseif(DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/Downloads/New_Project.wav")
  set(_VULK_TITLE_MENU_WAV "$ENV{HOME}/Downloads/New_Project.wav")
elseif(DEFINED ENV{USERPROFILE} AND EXISTS "$ENV{USERPROFILE}/Downloads/New_Project.wav")
  set(_VULK_TITLE_MENU_WAV "$ENV{USERPROFILE}/Downloads/New_Project.wav")
elseif(DEFINED ENV{HOME} AND EXISTS "$ENV{HOME}/Downloads/the_long_hall.wav")
  set(_VULK_TITLE_MENU_WAV "$ENV{HOME}/Downloads/the_long_hall.wav")
elseif(DEFINED ENV{USERPROFILE} AND EXISTS "$ENV{USERPROFILE}/Downloads/the_long_hall.wav")
  set(_VULK_TITLE_MENU_WAV "$ENV{USERPROFILE}/Downloads/the_long_hall.wav")
endif()
if(_VULK_TITLE_MENU_WAV)
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_TITLE_MENU_WAV=\"${_VULK_TITLE_MENU_WAV}\"")
  message(STATUS "vulkan_game: title menu WAV: ${_VULK_TITLE_MENU_WAV}")
endif()

if(EXISTS "${VULKAN_GAME_MESHY_HAIR_GLB}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_MESHY_HAIR_GLB=\"${VULKAN_GAME_MESHY_HAIR_GLB}\"")
  message(STATUS "vulkan_game: optional staff hair-shove GLB: ${VULKAN_GAME_MESHY_HAIR_GLB}")
endif()

# Staff / player H-key proximity dance: retarget Meshy hip-hop onto the idle rig; bundled proximity_dance.glb is fallback only.
# Cross-compiling Windows maps the chosen GLB under the repo to a portable path next to RetroIkea.exe when possible.
set(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH "" CACHE FILEPATH "GLB for staff/player proximity dance (override search path)")
set(_VULK_STAFF_DANCE_FN "proximity_dance.glb")
set(_VULK_STAFF_DANCE_SEARCH "")
if(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH)
  list(APPEND _VULK_STAFF_DANCE_SEARCH "${VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH}")
endif()
list(APPEND _VULK_STAFF_DANCE_SEARCH
  "${VULKAN_GAME_MESHY_REEXPORT_DIR}/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb"
  "${VULKAN_GAME_MESHY_REEXPORT_DIR}/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb")
list(APPEND _VULK_STAFF_DANCE_SEARCH "${CMAKE_CURRENT_SOURCE_DIR}/../Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{HOME}/Downloads/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{USERPROFILE}/Downloads/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
list(APPEND _VULK_STAFF_DANCE_SEARCH "${VULKAN_GAME_ASSETS_SOURCE_DIR}/models/character/${_VULK_STAFF_DANCE_FN}")
list(APPEND _VULK_STAFF_DANCE_SEARCH "${VULKAN_GAME_STAFF_MODEL_DIR}/${_VULK_STAFF_DANCE_FN}")
list(APPEND _VULK_STAFF_DANCE_SEARCH "${CMAKE_CURRENT_SOURCE_DIR}/../Downloads/${_VULK_STAFF_DANCE_FN}")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{HOME}/Downloads/${_VULK_STAFF_DANCE_FN}")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{USERPROFILE}/Downloads/${_VULK_STAFF_DANCE_FN}")
endif()
set(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND "")
foreach(_p IN LISTS _VULK_STAFF_DANCE_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND)
  vulkan_portable_data_path(_vulk_staff_dance_rt "${VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB=\"${_vulk_staff_dance_rt}\"")
  message(STATUS "vulkan_game: staff proximity dance GLB: ${_vulk_staff_dance_rt}")
endif()

# Rare easter egg: optional GLB + optional dance bed (MP3).
# Override: cmake -DVULKAN_GAME_SHREK_EGG_GLB_PATH=/path/to/shrek.glb
set(VULKAN_GAME_SHREK_EGG_GLB_PATH "" CACHE FILEPATH "Path to Shrek easter egg GLB (enables ] / F10 spawn when set and file exists)")
set(_VULK_SHREK_FN "proximity_dance.glb")
set(_VULK_SHREK_SEARCH "")
if(VULKAN_GAME_SHREK_EGG_GLB_PATH)
  list(APPEND _VULK_SHREK_SEARCH "${VULKAN_GAME_SHREK_EGG_GLB_PATH}")
endif()
list(APPEND _VULK_SHREK_SEARCH "${VULKAN_GAME_STAFF_MODEL_DIR}/${_VULK_SHREK_FN}")
list(APPEND _VULK_SHREK_SEARCH
  "${VULKAN_GAME_MESHY_REEXPORT_DIR}/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb")
list(APPEND _VULK_SHREK_SEARCH
  "${CMAKE_CURRENT_SOURCE_DIR}/../Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Meshy_AI_biped/${_VULK_SHREK_FN}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_SHREK_SEARCH "$ENV{HOME}/Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_SHREK_SEARCH "$ENV{USERPROFILE}/Downloads/Meshy_AI_biped/Meshy_AI_biped_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
set(VULKAN_GAME_SHREK_EGG_GLB_CAND "")
foreach(_p IN LISTS _VULK_SHREK_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_SHREK_EGG_GLB_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_SHREK_EGG_GLB_CAND)
  vulkan_portable_data_path(_vulk_shrek_egg_rt "${VULKAN_GAME_SHREK_EGG_GLB_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHREK_EGG_GLB=\"${_vulk_shrek_egg_rt}\"")
  message(STATUS "vulkan_game: Shrek easter egg GLB enabled: ${_vulk_shrek_egg_rt}")
else()
  message(STATUS "vulkan_game: Shrek GLB not found — ]/F10 spawn off. Add assets/models/character/proximity_dance.glb or set VULKAN_GAME_SHREK_EGG_GLB_PATH.")
endif()
set(_VULK_SHREK_MP3_FN "shrek_all_star.mp3")
set(_VULK_SHREK_MP3_SEARCH
  "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/${_VULK_SHREK_MP3_FN}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../All Star Low quality.mp3")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_SHREK_MP3_SEARCH "$ENV{HOME}/Downloads/All Star Low quality.mp3")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_SHREK_MP3_SEARCH "$ENV{USERPROFILE}/Downloads/All Star Low quality.mp3")
endif()
set(VULKAN_GAME_SHREK_EGG_MP3_CAND "")
foreach(_p IN LISTS _VULK_SHREK_MP3_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_SHREK_EGG_MP3_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_SHREK_EGG_MP3_CAND)
  set(_VULK_SHREK_MP3_RUNTIME "${VULKAN_GAME_SHREK_EGG_MP3_CAND}")
  if(VULKAN_GAME_SHREK_EGG_MP3_CAND STREQUAL "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/${_VULK_SHREK_MP3_FN}")
    set(_VULK_SHREK_MP3_RUNTIME "${VULKAN_GAME_ASSETS_DIR}/audio/${_VULK_SHREK_MP3_FN}")
  endif()
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHREK_EGG_MUSIC=\"${_VULK_SHREK_MP3_RUNTIME}\"")
  message(STATUS "vulkan_game: Shrek easter egg music: ${_VULK_SHREK_MP3_RUNTIME}")
endif()

# Optional deli prop: meatballs GLB for pickup/eat (same interaction flow as pizza).
set(VULKAN_GAME_DELI_MEATBALL_GLB_PATH "" CACHE FILEPATH "Path to deli meatballs GLB (optional)")
set(_VULK_DELI_MEATBALL_FN "Meshy_AI_sweetish_meat_balls_o_0404203406_texture.glb")
set(_VULK_DELI_MEATBALL_SEARCH "")
if(VULKAN_GAME_DELI_MEATBALL_GLB_PATH)
  list(APPEND _VULK_DELI_MEATBALL_SEARCH "${VULKAN_GAME_DELI_MEATBALL_GLB_PATH}")
endif()
list(APPEND _VULK_DELI_MEATBALL_SEARCH "${VULKAN_GAME_MESHY_REEXPORT_DIR}/${_VULK_DELI_MEATBALL_FN}")
set(VULKAN_GAME_DELI_MEATBALL_GLB_CAND "")
foreach(_p IN LISTS _VULK_DELI_MEATBALL_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_DELI_MEATBALL_GLB_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_DELI_MEATBALL_GLB_CAND)
  vulkan_portable_data_path(_vulk_deli_mb_rt "${VULKAN_GAME_DELI_MEATBALL_GLB_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_DELI_MEATBALL_GLB=\"${_vulk_deli_mb_rt}\"")
  message(STATUS "vulkan_game: deli meatballs GLB: ${_vulk_deli_mb_rt}")
else()
  message(STATUS "vulkan_game: deli meatballs GLB not found — add ${_VULK_DELI_MEATBALL_FN} to assets/meshy_reexport or set VULKAN_GAME_DELI_MEATBALL_GLB_PATH.")
endif()

# Box cutter + rusty pipe on shelf decks (enabled by default when bundled assets are present).
set(VULKAN_GAME_SHELF_DECK_PROPS ON CACHE BOOL "Instanced box cutter and rusty pipe on shelf decks" FORCE)

# Optional shelf prop: Meshy box cutter GLB (rarer spawn than ladders in-game).
set(VULKAN_GAME_SHELF_BOX_CUTTER_GLB_PATH "" CACHE FILEPATH "Path to box cutter GLB for shelf instancing")
set(_VULK_BOXCUT_FN "Meshy_AI_box_cutter_0402120433_texture.glb")
set(_VULK_BOXCUT_SEARCH "")
if(VULKAN_GAME_SHELF_BOX_CUTTER_GLB_PATH)
  list(APPEND _VULK_BOXCUT_SEARCH "${VULKAN_GAME_SHELF_BOX_CUTTER_GLB_PATH}")
endif()
list(APPEND _VULK_BOXCUT_SEARCH "${VULKAN_GAME_MESHY_REEXPORT_DIR}/${_VULK_BOXCUT_FN}")
list(APPEND _VULK_BOXCUT_SEARCH
  "${CMAKE_CURRENT_SOURCE_DIR}/../${_VULK_BOXCUT_FN}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Downloads/${_VULK_BOXCUT_FN}")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_BOXCUT_SEARCH "$ENV{HOME}/Downloads/${_VULK_BOXCUT_FN}")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_BOXCUT_SEARCH "$ENV{USERPROFILE}/Downloads/${_VULK_BOXCUT_FN}")
endif()
set(VULKAN_GAME_SHELF_BOX_CUTTER_GLB_CAND "")
foreach(_p IN LISTS _VULK_BOXCUT_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_SHELF_BOX_CUTTER_GLB_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_SHELF_DECK_PROPS AND VULKAN_GAME_SHELF_BOX_CUTTER_GLB_CAND)
  vulkan_portable_data_path(_vulk_box_rt "${VULKAN_GAME_SHELF_BOX_CUTTER_GLB_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHELF_BOX_CUTTER_GLB=\"${_vulk_box_rt}\"")
  message(STATUS "vulkan_game: shelf box cutter GLB: ${_vulk_box_rt}")
elseif(VULKAN_GAME_SHELF_BOX_CUTTER_GLB_CAND)
  message(STATUS "vulkan_game: shelf box cutter GLB found but disabled (set VULKAN_GAME_SHELF_DECK_PROPS=ON to enable).")
else()
  message(STATUS "vulkan_game: shelf box cutter GLB not found — optional prop disabled (set VULKAN_GAME_SHELF_BOX_CUTTER_GLB_PATH or place ${_VULK_BOXCUT_FN} in Downloads).")
endif()

# Optional shelf prop: rusty pipe GLB (more common in-game than the box cutter).
set(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_PATH "" CACHE FILEPATH "Path to rusty pipe GLB for shelf instancing")
set(_VULK_PIPE_FN "Meshy_AI_rusty_pipe_0402121715_texture.glb")
set(_VULK_PIPE_SEARCH "")
if(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_PATH)
  list(APPEND _VULK_PIPE_SEARCH "${VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_PATH}")
endif()
list(APPEND _VULK_PIPE_SEARCH "${VULKAN_GAME_MESHY_REEXPORT_DIR}/${_VULK_PIPE_FN}")
list(APPEND _VULK_PIPE_SEARCH
  "${CMAKE_CURRENT_SOURCE_DIR}/../${_VULK_PIPE_FN}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../Downloads/${_VULK_PIPE_FN}")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_PIPE_SEARCH "$ENV{HOME}/Downloads/${_VULK_PIPE_FN}")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_PIPE_SEARCH "$ENV{USERPROFILE}/Downloads/${_VULK_PIPE_FN}")
endif()
set(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_CAND "")
foreach(_p IN LISTS _VULK_PIPE_SEARCH)
  if(EXISTS "${_p}")
    set(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_CAND "${_p}")
    break()
  endif()
endforeach()
if(VULKAN_GAME_SHELF_DECK_PROPS AND VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_CAND)
  vulkan_portable_data_path(_vulk_pipe_rt "${VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHELF_RUSTY_PIPE_GLB=\"${_vulk_pipe_rt}\"")
  message(STATUS "vulkan_game: shelf rusty pipe GLB: ${_vulk_pipe_rt}")
elseif(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_CAND)
  message(STATUS "vulkan_game: shelf rusty pipe GLB found but disabled (set VULKAN_GAME_SHELF_DECK_PROPS=ON to enable).")
else()
  message(STATUS "vulkan_game: shelf rusty pipe GLB not found — optional prop disabled (set VULKAN_GAME_SHELF_RUSTY_PIPE_GLB_PATH or place ${_VULK_PIPE_FN} in Downloads).")
endif()

set(VULKAN_GAME_HEARTBEAT_MP3_CAND "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/ui_low_health_heartbeat.mp3")
if(NOT EXISTS "${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
  set(VULKAN_GAME_HEARTBEAT_MP3_CAND "${CMAKE_CURRENT_SOURCE_DIR}/../brvhrtz-heartbeat-02-225103.mp3")
endif()
if(EXISTS "${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
  set(_VULK_HEARTBEAT_MP3_RUNTIME "${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
  if(VULKAN_GAME_HEARTBEAT_MP3_CAND STREQUAL "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/ui_low_health_heartbeat.mp3")
    set(_VULK_HEARTBEAT_MP3_RUNTIME "${VULKAN_GAME_ASSETS_DIR}/audio/ui_low_health_heartbeat.mp3")
  endif()
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_HEARTBEAT_MP3=\"${_VULK_HEARTBEAT_MP3_RUNTIME}\"")
  message(STATUS "vulkan_game: low-health heartbeat MP3: ${_VULK_HEARTBEAT_MP3_RUNTIME}")
endif()

set(VULKAN_GAME_BIG_FALL_MP3_CAND "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/sfx_big_fall_impact.mp3")
if(NOT EXISTS "${VULKAN_GAME_BIG_FALL_MP3_CAND}")
  set(VULKAN_GAME_BIG_FALL_MP3_CAND "${CMAKE_CURRENT_SOURCE_DIR}/../universfield-falling-game-character-352287.mp3")
endif()
if(EXISTS "${VULKAN_GAME_BIG_FALL_MP3_CAND}")
  set(_VULK_BIG_FALL_MP3_RUNTIME "${VULKAN_GAME_BIG_FALL_MP3_CAND}")
  if(VULKAN_GAME_BIG_FALL_MP3_CAND STREQUAL "${VULKAN_GAME_ASSETS_SOURCE_DIR}/audio/sfx_big_fall_impact.mp3")
    set(_VULK_BIG_FALL_MP3_RUNTIME "${VULKAN_GAME_ASSETS_DIR}/audio/sfx_big_fall_impact.mp3")
  endif()
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_BIG_FALL_MP3=\"${_VULK_BIG_FALL_MP3_RUNTIME}\"")
  message(STATUS "vulkan_game: big fall impact MP3: ${_VULK_BIG_FALL_MP3_RUNTIME}")
endif()

# Optional runtime images: VULKAN_GAME_EXTRA_TEXTURES="a.png:b.png" (up to 16), optional
# VULKAN_GAME_EXTRA_BLEND=0..255 blends extraTex[0] onto default ground triplanar (default 0).
# Staff mesh: first three extras map to pants / shirt / skin when set; VULKAN_GAME_STAFF_TEX_BLEND=0..255 (default 255).
# Optional auto-load textures: assets/textures/staff_pants.png | staff_shirt.png | staff_skin.png (when present).
# Runtime: env VULKAN_GAME_STAFF_FORCE_PROCEDURAL_UNIFORM=1 uses yellow/blue IKEA shader instead of atlas from idle GLB.
option(VULKAN_GAME_STAFF_DISABLE_EMBEDDED_GLB_DIFFUSE
  "Use procedural IKEA staff uniform instead of embedded idle.glb diffuse (same as forcing env at runtime)"
  OFF)
# No default staff photo textures — NPCs use procedural uniform in shader.frag (empEmployee*).
# HUD TrueType font (crisp text vs stb_easy_font). Ship DejaVuSans.ttf under assets/fonts/ for all platforms.
# EXISTS must use VULKAN_GAME_ASSETS_SOURCE_DIR: on Windows installs VULKAN_GAME_ASSETS_DIR is only "assets"
# — CMake would look in the build dir and skip the compile definition despite the file being shipped.
if(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/fonts/DejaVuSans.ttf")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_UI_FONT_PATH=\"${VULKAN_GAME_ASSETS_DIR}/fonts/DejaVuSans.ttf\"")
  message(STATUS "vulkan_game: HUD UI font enabled: ${VULKAN_GAME_ASSETS_DIR}/fonts/DejaVuSans.ttf")
endif()
if(VULKAN_GAME_STAFF_DISABLE_EMBEDDED_GLB_DIFFUSE)
  target_compile_definitions(vulkan_game PRIVATE VULKAN_GAME_STAFF_DISABLE_EMBEDDED_DIFFUSE=1)
  message(STATUS "vulkan_game: staff procedural uniform (embedded GLB diffuse disabled at build time)")
endif()

# Scene textures load from disk at runtime under VULKAN_GAME_ASSETS_DIR (see main.cpp create*TextureResources).
# Optional override for deli brushed-metal. Do not fall back to shelf/wood textures; if the
# metal texture is absent, the procedural counter colors in main.cpp are used instead.
set(VULKAN_GAME_DELI_METAL_TEXTURE_PATH "" CACHE FILEPATH
  "Path to deli counter brushed-metal image (empty: pick from assets/textures/)")
if(VULKAN_GAME_DELI_METAL_TEXTURE_PATH)
  set(VULKAN_GAME_DELI_METAL_TEXTURE "${VULKAN_GAME_DELI_METAL_TEXTURE_PATH}")
elseif(EXISTS "${VULKAN_GAME_ASSETS_SOURCE_DIR}/textures/deli_metal_brushed.jpg")
  set(VULKAN_GAME_DELI_METAL_TEXTURE "${VULKAN_GAME_ASSETS_DIR}/textures/deli_metal_brushed.jpg")
else()
  set(VULKAN_GAME_DELI_METAL_TEXTURE "${VULKAN_GAME_ASSETS_DIR}/textures/deli_metal_brushed.jpg")
endif()
target_compile_definitions(vulkan_game PRIVATE
  "VULKAN_GAME_DELI_METAL_TEXTURE=\"${VULKAN_GAME_DELI_METAL_TEXTURE}\"")

set(_VULKAN_GAME_SDL2_IMAGE_LINK_TARGET "")
if(TARGET SDL2_image::SDL2_image)
  set(_VULKAN_GAME_SDL2_IMAGE_LINK_TARGET SDL2_image::SDL2_image)
elseif(TARGET SDL2_image::SDL2_image-static)
  set(_VULKAN_GAME_SDL2_IMAGE_LINK_TARGET SDL2_image::SDL2_image-static)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND
       EXISTS "/usr/x86_64-w64-mingw32/lib/libSDL2_image.dll.a")
  add_library(SDL2_image::SDL2_image UNKNOWN IMPORTED)
  set_target_properties(SDL2_image::SDL2_image PROPERTIES
    IMPORTED_LOCATION "/usr/x86_64-w64-mingw32/lib/libSDL2_image.dll.a"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/x86_64-w64-mingw32/include/SDL2")
  set(_VULKAN_GAME_SDL2_IMAGE_LINK_TARGET SDL2_image::SDL2_image)
endif()

if(NOT _VULKAN_GAME_SDL2_IMAGE_LINK_TARGET)
  message(FATAL_ERROR "SDL2_image imported target not found. Set SDL2_image_DIR to the MinGW SDL2_image config path.")
endif()

set(_VULKAN_GAME_SDL2_MAIN_LINK_TARGET "")
if((WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows") AND TARGET SDL2::SDL2main)
  set(_VULKAN_GAME_SDL2_MAIN_LINK_TARGET SDL2::SDL2main)
endif()

target_link_libraries(vulkan_game PRIVATE
  Vulkan::Vulkan
  ${_VULKAN_GAME_SDL2_MAIN_LINK_TARGET}
  SDL2::SDL2
  ${_VULKAN_GAME_SDL2_IMAGE_LINK_TARGET}
  assimp::assimp
  ReactPhysics3D::reactphysics3d
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
if(OpenSSL_FOUND)
  target_link_libraries(vulkan_game PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
if(WIN32)
  # crypt32: cpp-httplib HTTPS (OpenSSL) certificate loading uses Cert* APIs on Windows.
  target_link_libraries(vulkan_game PRIVATE ws2_32 iphlpapi crypt32)
endif()

# Arch MinGW SDL2_image static lib links IMG_webp.o; add WebP import libs for undefined WebP* / WebPDemux* symbols.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND
    EXISTS "/usr/x86_64-w64-mingw32/lib/libwebp.dll.a")
  target_link_libraries(vulkan_game PRIVATE
    /usr/x86_64-w64-mingw32/lib/libwebp.dll.a
    /usr/x86_64-w64-mingw32/lib/libwebpdemux.dll.a)
endif()
# Ship SDL2_image codec DLLs next to the .exe (PNG/JPEG/TIFF load at runtime via dlopen-style deps).
# WebP is explicit above; without zlib + libpng, IMG_Load of title/menu PNGs fails → solid IKEA-blue fallback UI.
# Inno Setup copies these from build-win-mingw next to RetroIkea.exe.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND
    EXISTS "/usr/x86_64-w64-mingw32/bin/libwebp.dll")
  add_custom_command(TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      /usr/x86_64-w64-mingw32/bin/libwebp.dll
      /usr/x86_64-w64-mingw32/bin/libwebpdemux.dll
      /usr/x86_64-w64-mingw32/bin/libsharpyuv.dll
      /usr/x86_64-w64-mingw32/bin/libssp-0.dll
      $<TARGET_FILE_DIR:vulkan_game>
    COMMENT "Copy MinGW WebP (and dependency) DLLs next to RetroIkea.exe")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND EXISTS "/usr/x86_64-w64-mingw32/bin/zlib1.dll")
  add_custom_command(TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      /usr/x86_64-w64-mingw32/bin/zlib1.dll
      $<TARGET_FILE_DIR:vulkan_game>
    COMMENT "Copy zlib for SDL_image PNG/TIFF")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND EXISTS "/usr/x86_64-w64-mingw32/bin/libpng16-16.dll")
  add_custom_command(TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      /usr/x86_64-w64-mingw32/bin/libpng16-16.dll
      $<TARGET_FILE_DIR:vulkan_game>
    COMMENT "Copy libpng for SDL_image PNG")
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND EXISTS "/usr/x86_64-w64-mingw32/bin/libjpeg-8.dll")
  add_custom_command(TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      /usr/x86_64-w64-mingw32/bin/libjpeg-8.dll
      $<TARGET_FILE_DIR:vulkan_game>
    COMMENT "Copy libjpeg for SDL_image JPG")
endif()
# Lobby HTTPS (cpp-httplib + OpenSSL): DLLs must sit next to RetroIkea.exe (Wine loads them like Windows).
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND OpenSSL_FOUND)
  set(_vk_mingw_ssl "")
  set(_vk_mingw_crypto "")
  if(VULKAN_GAME_MINGW_OPENSSL_SSL_DLL AND VULKAN_GAME_MINGW_OPENSSL_CRYPTO_DLL)
    set(_vk_mingw_ssl "${VULKAN_GAME_MINGW_OPENSSL_SSL_DLL}")
    set(_vk_mingw_crypto "${VULKAN_GAME_MINGW_OPENSSL_CRYPTO_DLL}")
  else()
    set(_vk_openssl_pairs
      "/usr/x86_64-w64-mingw32/bin/libssl-3-x64.dll:/usr/x86_64-w64-mingw32/bin/libcrypto-3-x64.dll"
      "/usr/x86_64-w64-mingw32/bin/libssl-3.dll:/usr/x86_64-w64-mingw32/bin/libcrypto-3.dll")
    foreach(_vk_pair IN LISTS _vk_openssl_pairs)
      string(REPLACE ":" ";" _vk_ab "${_vk_pair}")
      list(GET _vk_ab 0 _vk_s)
      list(GET _vk_ab 1 _vk_c)
      if(EXISTS "${_vk_s}" AND EXISTS "${_vk_c}")
        set(_vk_mingw_ssl "${_vk_s}")
        set(_vk_mingw_crypto "${_vk_c}")
        break()
      endif()
    endforeach()
  endif()
  if(_vk_mingw_ssl AND _vk_mingw_crypto)
    message(STATUS "vulkan_game: bundling OpenSSL DLLs: ${_vk_mingw_ssl} ${_vk_mingw_crypto}")
    # Always install as libssl-3-x64.dll / libcrypto-3-x64.dll (PE basename the exe imports + Inno lists).
    add_custom_command(TARGET vulkan_game POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_vk_mingw_ssl}"
        $<TARGET_FILE_DIR:vulkan_game>/libssl-3-x64.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_vk_mingw_crypto}"
        $<TARGET_FILE_DIR:vulkan_game>/libcrypto-3-x64.dll
      COMMENT "Copy OpenSSL DLLs for HTTPS lobby client")
  else()
    message(FATAL_ERROR
      "Windows cross-build links OpenSSL but MinGW OpenSSL DLLs were not found.\n"
      "Install mingw-w64 OpenSSL for your distro (DLLs in /usr/x86_64-w64-mingw32/bin/), or set\n"
      "  -DVULKAN_GAME_MINGW_OPENSSL_SSL_DLL=/path/to/libssl-*.dll\n"
      "  -DVULKAN_GAME_MINGW_OPENSSL_CRYPTO_DLL=/path/to/libcrypto-*.dll")
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  set(_vk_runtime_dll "${VULKAN_GAME_MINGW_VULKAN_DLL}")
  if(NOT _vk_runtime_dll)
    set(_vk_runtime_dll_candidates "")
    if(DEFINED ENV{VULKAN_SDK})
      list(APPEND _vk_runtime_dll_candidates
        "$ENV{VULKAN_SDK}/Bin/vulkan-1.dll"
        "$ENV{VULKAN_SDK}/Bin32/vulkan-1.dll")
    endif()
    list(APPEND _vk_runtime_dll_candidates
      "/usr/x86_64-w64-mingw32/bin/vulkan-1.dll")
    foreach(_vk_dll IN LISTS _vk_runtime_dll_candidates)
      if(EXISTS "${_vk_dll}")
        set(_vk_runtime_dll "${_vk_dll}")
        break()
      endif()
    endforeach()
  endif()
  if(_vk_runtime_dll)
    add_custom_command(TARGET vulkan_game POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_vk_runtime_dll}"
        $<TARGET_FILE_DIR:vulkan_game>/vulkan-1.dll
      COMMENT "Copy Vulkan loader DLL next to RetroIkea.exe")
  else()
    message(WARNING
      "vulkan-1.dll was not found for bundling. Set "
      "-DVULKAN_GAME_MINGW_VULKAN_DLL=/path/to/vulkan-1.dll to include it in the installer.")
  endif()
endif()

# MinGW default: embed libgcc/libstdc++/winpthread. VULKAN_GAME_WINDOWS_ALL_DYNAMIC ships those DLLs instead.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND NOT MSVC AND
   NOT VULKAN_GAME_WINDOWS_ALL_DYNAMIC)
  target_link_options(vulkan_game PRIVATE
    -static-libgcc
    -static-libstdc++
    -Wl,-Bstatic
    -lwinpthread
    -Wl,-Bdynamic
  )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND VULKAN_GAME_WINDOWS_ALL_DYNAMIC)
  add_custom_command(
    TARGET vulkan_game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:assimp::assimp>
            $<TARGET_FILE_DIR:vulkan_game>
    COMMENT "Copy libAssimp DLL next to RetroIkea.exe (VULKAN_GAME_WINDOWS_ALL_DYNAMIC)")
endif()
foreach(
  _dyn
  libgcc_s_seh-1.dll
  libstdc++-6.dll
  libwinpthread-1.dll)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND VULKAN_GAME_WINDOWS_ALL_DYNAMIC
     AND EXISTS "/usr/x86_64-w64-mingw32/bin/${_dyn}")
    add_custom_command(
      TARGET vulkan_game POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different /usr/x86_64-w64-mingw32/bin/${_dyn}
              $<TARGET_FILE_DIR:vulkan_game>
      COMMENT "Copy ${_dyn} (dynamic MinGW runtime)")
  endif()
endforeach()

if(MSVC)
  target_compile_options(vulkan_game PRIVATE /W4 /permissive-)
else()
  target_compile_options(vulkan_game PRIVATE -Wall -Wextra)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
    # MinGW libstdc++ can emit noisy narrowing diagnostics from <random> internals on some toolchains.
    target_compile_options(vulkan_game PRIVATE -Wno-narrowing)
  endif()
endif()
