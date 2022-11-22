# Face Morphing Installation

Computer vision powered art installation using face tracking and face morphing. Detects faces and landmarks from two webcam feeds. If one person's pupil locations match with the other person's, their faces morph into each other on both video feeds.

Uses OpenCV, dlib and threading to effectively locate and morph two faces. Threading means the landmark detection won't slow down the video feeds that are being displayed. 

Delaunay triangulation is used to create a mask out of each face and to do the morphing. OSC messages are sent to trigger soundscape changes in MaxMSP.

Presented for the first time at Corsica Studios in London with In-grid collective on 19 May 2022.&nbsp;  

**Installation and usage:**

Tested on Python 3.7.4. Make sure that you have two webcams available (for example, two USB webcams and the laptop lid closed to disable the laptop webcam).

```
git clone https://github.com/vjuss/Face-Morphing.git
cd Face-Morphing
pip3 install -r requirements.txt
python3 main.py
```
&nbsp;  
**Interaction:**

One person sits in front of each webcam. They see the face of the **other** person on their screen. Additionally, their **own** eyes are drawn on top of this video feed. With the drawn eyes as their guide, the two people are invited to align their eyes and form a connection. 

Once a connection is formed, the participants should maintain it for 30 seconds. During these 30 seconds, their faces are being morphed in varying degrees. This also affects the soundscape.