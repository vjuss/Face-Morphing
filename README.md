# Face-Morphing



Computer vision powered art installation using face tracking and face morphing. Presented for the first time at Corsica Studios with In-grid collective on 19 May 2022.

Uses OpenCV, dlib and threading to effectively locate and morph two faces - one on each webcam feed.
  <p>&nbsp;</p>


**Interaction:**

One person sits in front of each webcam. They see the face of the **other** person on their screen. Additionally, their **own** eyes are drawn on top of this video feed. With the drawn eyes as their guide, the two people are invited to align their faces and form a connection. 

Once a connection is formed, the participants should maintain it for 30 seconds. During these 30 seconds, their faces are being morphed in varying degrees.  
  <p>&nbsp;</p>


**Tech:**

OpenCV and threading to process and show two webcam feeds. 

dlib for locating faces and landmarks. 

Delaunay triangulation to create a mask out of each face and to do the morphing.

pythonosc to trigger sound in MaxMSP.
