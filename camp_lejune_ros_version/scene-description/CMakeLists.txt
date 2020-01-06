cmake_minimum_required(VERSION 2.8.3)
project(captioner)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  message_generation
#  catkin_virtualenv
)


catkin_python_setup()

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   ModuleMessage.msg
 )

## Generate services in the 'srv' folder
add_service_files(
   FILES
   SceneDescription.srv
 )


## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   #ModuleMessage
   sensor_msgs#   std_msgs
   #std_msgs
 )

catkin_package( 
  CATKIN_DEPENDS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  message_generation
  )


# install(
#   DIRECTORY
#   src
#   models
#   launch
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#   USE_SOURCE_PERMISSIONS
#   )


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES captioner
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs message_runtime
#  DEPENDS system_lib
)

# Use catkin_virtualenv to install pip deps locally
# catkin_generate_virtualenv()


catkin_install_python(
  PROGRAMS
  nodes/ros_caption.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )


install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
  )

install(DIRECTORY models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
  )


