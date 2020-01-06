#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
#from module_message_msgs.msg import ModuleMessage
#from module_message_msgs.srv import *
from captioner.srv import *
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from captioner.ros_caption_model2 import model as Model
from captioner.utils import decode_captions

class RosTensorFlow():
    def __init__(self):
        self._session = tf.Session()
        self.model = Model()
        self.captions = self.model.build_sampler()
        rospy.loginfo('build sucess')
        self._session.run(tf.global_variables_initializer())
        self.model.init(self._session)
        rospy.loginfo('init sucess')
        self._cv_bridge = CvBridge()
        #self._sub = rospy.Subscriber('image_raw', Image, self.callback, queue_size=1) # change image topic
        #self.msg = module_message_msgs.msg.ModuleMessage()
        self.service = rospy.Service('scene_description2',SceneDescription, self.handle_scene_description)
        #self._sub = rospy.Subscriber('AdonisRos/image', Image, self.callback)
        #self._pub = rospy.Publisher('/publish_message', ModuleMessage,queue_size=10)
    
    def handle_scene_description(self,req):
        image_msg = rospy.wait_for_message('image_raw',Image)
        now = rospy.get_time()
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        image_data = cv2.imencode('.jpg', cv_image)[1].tostring()
        generated_captions = self._session.run(
           self.captions, {self.model.image: cv_image})
        decoded_caption = decode_captions(generated_captions, self.model.idx_to_word)
        caption = decoded_caption[0].replace('with a building','')
        caption = caption.replace('and a building','')
        return SceneDescriptionResponse(now,caption)


    # def callback(self, image_msg):
    #     #rospy.wait_for_service('publish_message')
    #     cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
    #     image_data = cv2.imencode('.jpg', cv_image)[1].tostring()
    #     generated_captions = self._session.run(
    #        self.captions, {self.model.image: cv_image})
    #     decoded_caption = decode_captions(generated_captions, self.model.idx_to_word)
    #     self.msg.plain_text = decoded_caption[0]
    #     self.msg.destination_type = ['MMI']
    #     self.msg.origin_type = 'CMUC'
    #     self._pub.publish(self.msg)
    #     #rospy.loginfo(decoded_caption[0])
    #     # publish_message = rospy.ServiceProxy('publish_message', PublishMessage)
    #     # message_id = publish_message('CMUC', self.msg)
    #     # rospy.loginfo(message_id)
    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('CMU_captioner2')
    tensor = RosTensorFlow()
    tensor.main()
