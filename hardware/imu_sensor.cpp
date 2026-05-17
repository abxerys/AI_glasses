#include <stdint.h>

namespace imu_sensor {

struct ImuSample {
  float acc_x;
  float acc_y;
  float acc_z;
  float gyro_x;
  float gyro_y;
  float gyro_z;
};

bool initImu() {
  // TODO: Initialize IMU over I2C/SPI.
  return true;
}

ImuSample readImuData() {
  // TODO: Replace with real sensor read logic.
  return {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F};
}

bool detectFall(const ImuSample& sample) {
  // TODO: Improve threshold and temporal model.
  const float impact_threshold = 2.5F;
  return (sample.acc_x * sample.acc_x + sample.acc_y * sample.acc_y + sample.acc_z * sample.acc_z) >
         (impact_threshold * impact_threshold);
}

}  // namespace imu_sensor
