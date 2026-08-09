cmake_minimum_required(VERSION 3.8)
project(supermarket_scene_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Slot.msg"
  "msg/SlotPose.msg"
  "msg/DetectedBox.msg"
  "msg/InspectionSlot.msg"
  "srv/InspectShelf.srv"
  "srv/FindSku.srv"
  "srv/GetSlotPose.srv"
  "srv/GetSkuLocations.srv"
  "srv/ListSlots.srv"
  "srv/TakeSlot.srv"
  "srv/RestockSlot.srv"
  "srv/UpdateSlot.srv"
  "srv/CreateShelf.srv"
  "srv/UpdateShelf.srv"
  "srv/CreateDeliveryTable.srv"
  "srv/UpdateDeliveryTable.srv"
  "srv/CreateSlot.srv"
  DEPENDENCIES sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
