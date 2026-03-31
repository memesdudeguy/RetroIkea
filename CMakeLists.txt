cmake_minimum_required(VERSION 3.16)
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
# MinGW Windows builds: link Assimp statically so libassimp-5.dll is not required at runtime.
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  message(STATUS "vulkan_game: Windows cross-build — BUILD_SHARED_LIBS=OFF for static Assimp")
endif()
FetchContent_MakeAvailable(assimp)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
  set(VULKAN_CROSS_LIBRARY "" CACHE FILEPATH "Path to Windows Vulkan import library (vulkan-1.lib or libvulkan-1.a)")
  set(VULKAN_CROSS_INCLUDE_DIR "" CACHE PATH "Path to Vulkan headers (directory containing vulkan/vulkan.h)")
  # MinGW cross builds often need an explicit Windows Vulkan import library path.
  set(_VULKAN_CROSS_HINTS "")
  if(DEFINED ENV{VULKAN_SDK})
    list(APPEND _VULKAN_CROSS_HINTS
      "$ENV{VULKAN_SDK}/Lib"
      "$ENV{VULKAN_SDK}/lib"
      "$ENV{VULKAN_SDK}/Bin"
      "$ENV{VULKAN_SDK}/bin")
  endif()
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
find_package(Python3 3.6 COMPONENTS Interpreter REQUIRED)

find_program(GLSL_COMPILER glslc
  HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
  DOC "Khronos glslc (Vulkan SDK); required to compile shaders")
if(NOT GLSL_COMPILER)
  message(FATAL_ERROR "glslc not found. Install Vulkan SDK or set GLSL_COMPILER to the glslc path.")
endif()
set(VULKAN_GAME_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(SHADER_DIR ${VULKAN_GAME_ASSETS_DIR}/shaders)
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
)

if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set_target_properties(vulkan_game PROPERTIES OUTPUT_NAME RetroIkea)
endif()

add_dependencies(vulkan_game shaders)

target_include_directories(vulkan_game PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${glm_SOURCE_DIR}
)

# All shipped art/audio under vulkan_game/assets/ (see assets/models, assets/audio, assets/textures).
set(STORE_AMBIENT_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/store_ambient_loop.wav")
set(POWER_SWITCH_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/power_switch.mp3")
set(HORROR_AMBIENT_MP3 "${VULKAN_GAME_ASSETS_DIR}/audio/horror_ambient.mp3")
set(CHASE_MUSIC_MP3 "${VULKAN_GAME_ASSETS_DIR}/audio/chase_music.mp3")
set(LIGHT_SWITCH_ON_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/light_switch_on.mp3")
set(STAFF_SPOTTED_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/staff_spotted.mp3")
set(STAFF_CHASE_VO_SFX "${VULKAN_GAME_ASSETS_DIR}/audio/staff_chase_vo.mp3")

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
set(VULKAN_GAME_JUMPING_FBX "${VULKAN_GAME_STAFF_MODEL_DIR}/jump_regular.glb")
set(VULKAN_GAME_JUMP_RUN_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/jump_run.glb")
set(VULKAN_GAME_MESHY_PUNCH_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/punch_combo.glb")
set(VULKAN_GAME_MESHY_FALL_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/fall_knockdown.glb")
set(VULKAN_GAME_MESHY_STANDUP_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/stand_up.glb")
set(VULKAN_GAME_MESHY_LAND_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/land_hit_reaction.glb")
set(VULKAN_GAME_MESHY_HAIR_GLB "${VULKAN_GAME_STAFF_MODEL_DIR}/hair_shove_reaction.glb")
if(EXISTS "${VULKAN_GAME_MESHY_IDLE_GLB}")
  set(VULKAN_GAME_EMPLOYEE_FBX "${VULKAN_GAME_MESHY_IDLE_GLB}")
  message(STATUS "vulkan_game staff mesh: ${VULKAN_GAME_MESHY_IDLE_GLB}")
else()
  set(VULKAN_GAME_EMPLOYEE_FBX "${VULKAN_GAME_ASSETS_DIR}/models/character_base.fbx")
  message(STATUS "vulkan_game staff mesh: fallback ${VULKAN_GAME_EMPLOYEE_FBX}")
endif()

target_compile_definitions(vulkan_game PRIVATE
  "VULKAN_GAME_VERSION_STRING=\"${PROJECT_VERSION}\""
  SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
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
  "VULKAN_GAME_JUMPING_FBX=\"${VULKAN_GAME_JUMPING_FBX}\""
  "VULKAN_GAME_JUMP_RUN_GLB=\"${VULKAN_GAME_JUMP_RUN_GLB}\""
  "VULKAN_GAME_MESHY_PUNCH_GLB=\"${VULKAN_GAME_MESHY_PUNCH_GLB}\""
  "VULKAN_GAME_MESHY_FALL_GLB=\"${VULKAN_GAME_MESHY_FALL_GLB}\""
  "VULKAN_GAME_MESHY_STANDUP_GLB=\"${VULKAN_GAME_MESHY_STANDUP_GLB}\""
  "VULKAN_GAME_MESHY_LAND_GLB=\"${VULKAN_GAME_MESHY_LAND_GLB}\""
  "VULKAN_GAME_STAFF_MELEE_IMPACT_MP3=\"${VULKAN_GAME_ASSETS_DIR}/audio/sfx_staff_melee_impact.mp3\""
)

# Title screen loop (default: the_long_hall.wav in assets or ~/Downloads).
set(_VULK_TITLE_MENU_WAV "")
if(EXISTS "${VULKAN_GAME_ASSETS_DIR}/audio/the_long_hall.wav")
  set(_VULK_TITLE_MENU_WAV "${VULKAN_GAME_ASSETS_DIR}/audio/the_long_hall.wav")
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

# Staff / player dance near Shrek: retargeted onto idle rig from a Meshy hip-hop GLB (same clip for both).
# Prefer Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb in Downloads; fallback: bundled proximity_dance.glb.
set(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH "" CACHE FILEPATH "GLB for staff/player proximity dance (override search path)")
set(_VULK_STAFF_DANCE_FN "proximity_dance.glb")
set(_VULK_STAFF_DANCE_SEARCH "")
if(VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH)
  list(APPEND _VULK_STAFF_DANCE_SEARCH "${VULKAN_GAME_STAFF_SHREK_DANCE_GLB_PATH}")
endif()
# Hip-hop source (user-requested) before bundled character/proximity_dance.glb so staff + H emote match this motion.
list(APPEND _VULK_STAFF_DANCE_SEARCH "${CMAKE_CURRENT_SOURCE_DIR}/../Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
if(DEFINED ENV{HOME})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{HOME}/Downloads/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
if(DEFINED ENV{USERPROFILE})
  list(APPEND _VULK_STAFF_DANCE_SEARCH "$ENV{USERPROFILE}/Downloads/Meshy_AI_Animation_Hip_Hop_Dance_3_withSkin.glb")
endif()
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
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB=\"${VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND}\"")
  message(STATUS "vulkan_game: staff Shrek proximity dance GLB: ${VULKAN_GAME_STAFF_SHREK_DANCE_GLB_CAND}")
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
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHREK_EGG_GLB=\"${VULKAN_GAME_SHREK_EGG_GLB_CAND}\"")
  message(STATUS "vulkan_game: Shrek easter egg GLB enabled: ${VULKAN_GAME_SHREK_EGG_GLB_CAND}")
else()
  message(STATUS "vulkan_game: Shrek GLB not found — ]/F10 spawn off. Add assets/models/character/proximity_dance.glb or set VULKAN_GAME_SHREK_EGG_GLB_PATH.")
endif()
set(_VULK_SHREK_MP3_FN "shrek_all_star.mp3")
set(_VULK_SHREK_MP3_SEARCH
  "${VULKAN_GAME_ASSETS_DIR}/audio/${_VULK_SHREK_MP3_FN}"
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
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_SHREK_EGG_MUSIC=\"${VULKAN_GAME_SHREK_EGG_MP3_CAND}\"")
  message(STATUS "vulkan_game: Shrek easter egg music: ${VULKAN_GAME_SHREK_EGG_MP3_CAND}")
endif()

set(VULKAN_GAME_HEARTBEAT_MP3_CAND "${VULKAN_GAME_ASSETS_DIR}/audio/ui_low_health_heartbeat.mp3")
if(NOT EXISTS "${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
  set(VULKAN_GAME_HEARTBEAT_MP3_CAND "${CMAKE_CURRENT_SOURCE_DIR}/../brvhrtz-heartbeat-02-225103.mp3")
endif()
if(EXISTS "${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_HEARTBEAT_MP3=\"${VULKAN_GAME_HEARTBEAT_MP3_CAND}\"")
  message(STATUS "vulkan_game: low-health heartbeat MP3: ${VULKAN_GAME_HEARTBEAT_MP3_CAND}")
endif()

set(VULKAN_GAME_BIG_FALL_MP3_CAND "${VULKAN_GAME_ASSETS_DIR}/audio/sfx_big_fall_impact.mp3")
if(NOT EXISTS "${VULKAN_GAME_BIG_FALL_MP3_CAND}")
  set(VULKAN_GAME_BIG_FALL_MP3_CAND "${CMAKE_CURRENT_SOURCE_DIR}/../universfield-falling-game-character-352287.mp3")
endif()
if(EXISTS "${VULKAN_GAME_BIG_FALL_MP3_CAND}")
  target_compile_definitions(vulkan_game PRIVATE
    "VULKAN_GAME_BIG_FALL_MP3=\"${VULKAN_GAME_BIG_FALL_MP3_CAND}\"")
  message(STATUS "vulkan_game: big fall impact MP3: ${VULKAN_GAME_BIG_FALL_MP3_CAND}")
endif()

# Optional runtime images: VULKAN_GAME_EXTRA_TEXTURES="a.png:b.png" (up to 16), optional
# VULKAN_GAME_EXTRA_BLEND=0..255 blends extraTex[0] onto default ground triplanar (default 0).
# Staff mesh: first three extras map to pants / shirt / skin when set; VULKAN_GAME_STAFF_TEX_BLEND=0..255 (default 255).
# No default staff photo textures — NPCs use procedural uniform in shader.frag (empEmployee*).
# HUD TrueType font (crisp text vs stb_easy_font). Ship DejaVuSans.ttf under assets/fonts/ for all platforms.
set(VULKAN_GAME_UI_FONT "${VULKAN_GAME_ASSETS_DIR}/fonts/DejaVuSans.ttf")
if(EXISTS "${VULKAN_GAME_UI_FONT}")
  target_compile_definitions(vulkan_game PRIVATE "VULKAN_GAME_UI_FONT_PATH=\"${VULKAN_GAME_UI_FONT}\"")
endif()

# Embed runtime textures / SFX directly into the executable (paths must exist at configure time).
set(SCENE_TEXTURE_ASSET "${VULKAN_GAME_ASSETS_DIR}/textures/wall_concrete_gray.png")
set(SIGN_TEXTURE_ASSET "${VULKAN_GAME_ASSETS_DIR}/textures/sign.png")
set(SHELF_TEXTURE_ASSET "${VULKAN_GAME_ASSETS_DIR}/textures/shelf_rack_metal_osb.jpg")
set(CRATE_TEXTURE_ASSET "${VULKAN_GAME_ASSETS_DIR}/textures/crate_cardboard.jpeg")
set(FOOTSTEP_SOUND_ASSET "${VULKAN_GAME_ASSETS_DIR}/audio/sfx_footstep_concrete.mp3")
set(SLIDE_SOUND_ASSET "${VULKAN_GAME_ASSETS_DIR}/audio/sfx_slide_body_fall.mp3")

set(EMBEDDED_ASSETS_HEADER "${CMAKE_CURRENT_BINARY_DIR}/vulkan_embedded_assets.h")
set(EMBED_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/embed_assets_manifest.txt")
set(EMBED_ALL_PY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_all_assets.py")
# One Python process writes the whole header (sequential CMake execute_process runs interleave badly).
file(WRITE "${EMBED_MANIFEST}" "")
function(embed_manifest_line SYMBOL PATH)
  file(APPEND "${EMBED_MANIFEST}" "${SYMBOL}\t${PATH}\n")
endfunction()
embed_manifest_line(kSceneTextureData "${SCENE_TEXTURE_ASSET}")
embed_manifest_line(kSignTextureData "${SIGN_TEXTURE_ASSET}")
embed_manifest_line(kShelfRackTextureData "${SHELF_TEXTURE_ASSET}")
embed_manifest_line(kCrateTextureData "${CRATE_TEXTURE_ASSET}")
embed_manifest_line(kFootstepSoundData "${FOOTSTEP_SOUND_ASSET}")
embed_manifest_line(kSlideSoundData "${SLIDE_SOUND_ASSET}")
execute_process(
  COMMAND ${Python3_EXECUTABLE} "${EMBED_ALL_PY}" "${EMBEDDED_ASSETS_HEADER}" "${EMBED_MANIFEST}"
  RESULT_VARIABLE _embed_rv)
if(NOT _embed_rv EQUAL 0)
  message(FATAL_ERROR "embed_all_assets.py failed (${_embed_rv})")
endif()

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
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

# MinGW: embed libgcc/libstdc++/winpthread so the .exe runs without MinGW DLLs next to it (Wine/real Windows).
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING AND NOT MSVC)
  target_link_options(vulkan_game PRIVATE
    -static-libgcc
    -static-libstdc++
    -Wl,-Bstatic
    -lwinpthread
    -Wl,-Bdynamic
  )
endif()

if(MSVC)
  target_compile_options(vulkan_game PRIVATE /W4 /permissive-)
else()
  target_compile_options(vulkan_game PRIVATE -Wall -Wextra)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_CROSSCOMPILING)
    # MinGW libstdc++ can emit noisy narrowing diagnostics from <random> internals on some toolchains.
    target_compile_options(vulkan_game PRIVATE -Wno-narrowing)
  endif()
endif()
