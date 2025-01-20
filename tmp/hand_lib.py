# pip install pyserial
# ls /dev/ttyUSB1
# sudo chmod 666 /dev/ttyUSB1

import serial
import binascii
# import random

class Hand_Device(object):
    def __init__(self) -> None:
        self.hand_id = 1 # default 1, could be 0-256
        self.port = "/dev/ttyUSB0"
        self.baud = 115200 # 115200bps

        self.serial = serial.Serial()
        self.serial.port = self.port
        self.serial.baudrate = self.baud
        self.serial.open()
        
        self.finger_sim_angle_range = [0, 85]
        self.thumb_yaw_sim_angle_range = [0, 75]
        self.thumb_pitch_sim_angle_range = [-5, 35]
        
    def hand_serial_servo_write6_pos(self, positions):
        pos1, pos2, pos3, pos4, pos5, pos6 = positions
        
        value1_H = pos1 & 0xff
        value1_L = (pos1 >> 8) & 0xff
        
        value2_H = pos2 & 0xff
        value2_L = (pos2 >> 8) & 0xff
        
        value3_H = pos3 & 0xff
        value3_L = (pos3 >> 8) & 0xff
        
        value4_H = pos4 & 0xff
        value4_L = (pos4 >> 8) & 0xff
        
        value5_H = pos5 & 0xff
        value5_L = (pos5 >> 8) & 0xff
        
        value6_H = pos6 & 0xff
        value6_L = (pos6 >> 8) & 0xff
        
        # send command to set pos
        packet = bytearray()
        packet.append(0xEB)
        packet.append(0x90)
        packet.append(self.hand_id)
        # data length
        packet.append(0x0F)
        # Command get state
        packet.append(0x12)
        packet.append(0xC2)
        packet.append(0x05)

        packet.append(value1_H)
        packet.append(value1_L)
        packet.append(value2_H)
        packet.append(value2_L)
        packet.append(value3_H)
        packet.append(value3_L)
        packet.append(value4_H)
        packet.append(value4_L)
        packet.append(value5_H)
        packet.append(value5_L)
        packet.append(value6_H)
        packet.append(value6_L)
        
        checksum = self._calculate_checknum(packet)
        packet.append(checksum)
        try:
            self.serial.write(packet)
            # response = self.serial.read(64)
            # print(binascii.hexlify(response))
        except:
            print("hand_serial_servo_write6 rs485 error")
            self.serial.close()
        
    def hand_serial_servo_write6_speed(self, speeds):
        speed1, speed2, speed3, speed4, speed5, speed6 = speeds
        
        packet = bytearray()
        packet.append(0xEB)
        packet.append(0x90)
        packet.append(self.hand_id)
        # data length
        packet.append(0x0F)
        packet.append(0x12)
        packet.append(0xF2)
        packet.append(0x05)
        
        value1_H = speed1 & 0xff
        value1_L = (speed1 >> 8) & 0xff
        
        value2_H = speed2 & 0xff
        value2_L = (speed2 >> 8) & 0xff
        
        value3_H = speed3 & 0xff
        value3_L = (speed3 >> 8) & 0xff
        
        value4_H = speed4 & 0xff
        value4_L = (speed4 >> 8) & 0xff
        
        value5_H = speed5 & 0xff
        value5_L = (speed5 >> 8) & 0xff
        
        value6_H = speed6 & 0xff
        value6_L = (speed6 >> 8) & 0xff
        
        packet.append(value1_H)
        packet.append(value1_L)
        packet.append(value2_H)
        packet.append(value2_L)
        packet.append(value3_H)
        packet.append(value3_L)
        packet.append(value4_H)
        packet.append(value4_L)
        packet.append(value5_H)
        packet.append(value5_L)
        packet.append(value6_H)
        packet.append(value6_L)
        
        checksum = self._calculate_checknum(packet)
        packet.append(checksum)
        try:
            self.serial.write(packet)
            # response = self.serial.read(64)
            # print(binascii.hexlify(response))
        except:
            print("hand_serial_servo_write6 rs485 error")
            self.serial.close()
        
    def hand_serial_servo_write6_sim_angle(self, angles):
        """Set joints' sim angles"""
        angle1, angle2, angle3, angle4, angle5, angle6 = angles
        # TODO: check angle range here
        
        # int.tobytes big
        # 0.pinky(5) 1.ring(4) 2. middle(3) 3. index(2) 4. thumb(1) pitch 5. thumb(1) yaw
        
        pos1 = self._sim2real_finger_pos(angle1)
        pos2 = self._sim2real_finger_pos(angle2)
        pos3 = self._sim2real_finger_pos(angle3)
        pos4 = self._sim2real_finger_pos(angle4)
        pos5 = self._sim2real_thumb_pitch_pos(angle5)
        pos6 = self._sim2real_thumb_yaw_pos(angle6)
        print(f"{pos1} {pos2} {pos3} {pos4} {pos5} {pos6}")
        
        self.hand_serial_servo_write6_pos((pos1, pos2, pos3, pos4, pos5, pos6))
        
    # def randomize_sim_angles(self):
    #     index = random.randint(self.finger_sim_angle_range[0], self.finger_sim_angle_range[1])
    #     middle = random.randint(self.finger_sim_angle_range[0], self.finger_sim_angle_range[1])
    #     ring = random.randint(self.finger_sim_angle_range[0], self.finger_sim_angle_range[1])
    #     pinky = random.randint(self.finger_sim_angle_range[0], self.finger_sim_angle_range[1])
    #     thumb_yaw = random.randint(self.thumb_yaw_sim_angle_range[0], self.thumb_yaw_sim_angle_range[1])
    #     thumb_pitch = random.randint(self.thumb_pitch_sim_angle_range[0], self.thumb_pitch_sim_angle_range[1])
    #     return (pinky, ring, middle, index, thumb_yaw, thumb_pitch)

    def center_sim_angles(self):
        return (0, 0, 0, 0, 0, 0)
    
    def _sim2real_finger_pos(self, angle):
        """[0, 85] in modeing = [0, 2000] pos in real"""        
        angle_L = self.finger_sim_angle_range[0] # pos_L's angle
        angle_H = self.finger_sim_angle_range[1] # pos_H's angle
        
        pos_L = 0
        pos_H = 2000
        
        pos = int(((angle -angle_L)/(angle_H - angle_L))*(pos_H - pos_L) + pos_L)
        return pos
        
    def _sim2real_thumb_yaw_pos(self, angle):
        """[0, 75] in modeling = [0, 2000] in real"""
        angle_L = self.thumb_yaw_sim_angle_range[0] # pos_L's angle
        angle_H = self.thumb_yaw_sim_angle_range[1] # pos_H's angle
        
        pos_L = 0
        pos_H = 2000
        
        pos = int(((angle -angle_L)/(angle_H - angle_L))*(pos_H - pos_L) + pos_L)
        return pos
        
    def _sim2real_thumb_pitch_pos(self, angle):
        """[-5, 35] in modeling = [0, 2000] in real"""
        angle_L = self.thumb_pitch_sim_angle_range[0] # pos_L's angle
        angle_H = self.thumb_pitch_sim_angle_range[1] # pos_H's angle
        
        pos_L = 0
        pos_H = 2000
        
        pos = int(((angle -angle_L)/(angle_H - angle_L))*(pos_H - pos_L) + pos_L)
        return pos
            
    
    # def _sim2real_offset(self, angle, reverse, offset):
    #     if reverse:
    #         angle = 180 - angle
    #     angle = angle+offset
    #     return angle
        
    def _calculate_checknum(self, packet):
        check_num = 0
        length = int(packet[3])+5
        
        for i in range(2, length-1):
            check_num =check_num+packet[i]
        check_num = check_num & 0xff
        return check_num
    
    def _packet_add_checksum(self, packet: bytearray):
        check_num = self._calculate_checknum(packet)
        packet.append(check_num)
        return packet
        

# packet = bytearray()
# packet.append(0xEB)
# packet.append(0x90)
# packet.append(hand_id)
# packet.append(0x04)
# packet.append(0x11)
# packet.append(0xFE)
# packet.append(0x05)
# packet.append(0x0C)

# check_num = 0
# # length = int("04", 16)+5
# length = int(packet[3])+5
# print(length)
# for i in range(2, length-1):
#     check_num =check_num+packet[i]
# check_num = check_num & 0xff
# packet.append(check_num)
# # command = binascii.unhexlify(f"EB90{hand_id}0411FE050C")+check_num
# serial.write(packet)

# response = serial.read(64)
# print(binascii.hexlify(response))
# response = serial.readline()
# print(response)

# hand = Hand_Device()

# hand.hand_serial_servo_write6_speed((100, 100, 100, 100, 100, 100))
# import time
# time.sleep(1)
# hand.hand_serial_servo_write6_sim_angle(hand.randomize_sim_angles())
# hand.hand_serial_servo_write6_sim_angle(hand.center_sim_angles())
# hand.hand_serial_servo_write6((fingle_angle, fingle_angle, fingle_angle, fingle_angle, 0, 35))
# hand.hand_serial_servo_write6_pos((2000, 0, 0, 0, 0, 0))
