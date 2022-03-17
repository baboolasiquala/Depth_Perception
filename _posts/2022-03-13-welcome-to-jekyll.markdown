---
layout: post
author: "Digjay Das, Ivan Stepanov and Shaheryar Hasnain"
title:  "CSE 455 Depth Perception Project"
date:   2022-03-10 15:33:51 -0800
categories: jekyll update
permalink: "CSE455 Project"
---
This Blog is meant to demonstrate how Shaheryar Hasnain(), Ivan Stepanov(), Digjay Das (2029455) used their knowledge of Computer Vision in order to create a map of how far an object is with respect to the image produced by the cameras. 

In order to deduce depth perception we are in need of 2 cameras at a specified horizontal distance to each other and are relatively on the same vertical axis. Both cameras are required to take pictures at the same time of the same scene in order to deduce depth. It does so by using Stereo Disparity, which is the difference in image location from the left and right image taken from the camera. 

The rationale provided above is also the reason why a single camera is not sufficient to deduce depth, at least without being aided with the help of neural networks or using Active stereo with the employment of light.

In order to use the Disparity to deduce depth we have to then use Traingulation which is a geometric approach used to deduce depth. 

In order to go about this math we must understand that there are 3 planes we travel in between in order to deduce depth. There is the world co-ordinate plane, the camera co-ordinate plane and the image co-ordinate plane. Image plane is where the image sensor lies and the camera plane is where the lens lies. A transformation is undergone from 3D to 2D when we tranform the co-ordinates from the world co-ordinates to the image co-ordinates and vice versa. The transformation from the world co-ordinates to the camera co-ordinates is a 3-D to 3-D tranformation and the tranformation from the camera to the image plane in a 3-D to 2-D transformation. 

<h2>Camera Calibration</h2>

Based on our working understanding of perspective projections we can infer the image co-ordinates with respect to the camera.<br><br>
![1](/assets/images/1.png){:class="img-responsive"}<br><br>
![2](/assets/images/2.png){:class="img-responsive"}<br>
![3](/assets/images/3.png){:class="img-responsive"}<br><br>
![4](/assets/images/4.png){:class="img-responsive"}<br>

Bearing in mind that the image co-ordinates are captures by the image sensors of the cameras, we need to transform the co-ordinates once more from standard co-ordinates to pixels. When doing that we realize that the pixels themselves need not necessarily be square in nature but may also be rectangular.

The principal is with respect to the top left corner of an image. <br><br>
![5](/assets/images/5.png){:class="img-responsive"}<br>
![6](/assets/images/6.png){:class="img-responsive"}<br>
![7](/assets/images/7.png){:class="img-responsive"}<br>

The directional focal lengths and the principal point is referred to as the camera's internal geometry, thus bearing out the intrinsic matrix.

Once we gain the 2D parameters of the image we have convert it into its homogenous 3D representation in order to do the transformation into the camera's co-ordinate system.<br><br>
![8](/assets/images/8.png){:class="img-responsive"}<br>
![9](/assets/images/9.png){:class="img-responsive"}<br>
![10](/assets/images/10.png){:class="img-responsive"}<br>
![11](/assets/images/11.png){:class="img-responsive"}<br>

Now we have to do the mapping from camera co-ordinate frame to the world co-ordinate frame which is a 3D to 3D transformation, this done knowing the position and orientation of the camera co-ordinate frame with respect to the world co-ordinate frame. <br><br>

![12](/assets/images/12.png){:class="img-responsive"}<br><br>
It should be noted that the rotation matrix is a orthonormal matrix as in, when a dot product is carried by itself it produces an Identity matrix and it's inverse is equivalent to it's transpose.<br><br>

![13](/assets/images/13.png){:class="img-responsive"}<br><br>
![13_1](/assets/images/13_1.png){:class="img-responsive"}<br>
![14](/assets/images/14.png){:class="img-responsive"}<br><br>
![15](/assets/images/15.png){:class="img-responsive"}<br>


In order to get a more concise algorithm to solve things, we can transform our current matrix into Homogenous Co-ordinates.<br><br>

![16](/assets/images/16.png){:class="img-responsive"}<br><br>
![17](/assets/images/17.png){:class="img-responsive"}<br><br>
![18](/assets/images/18.png){:class="img-responsive"}<br>

As noted from above we see that we can extract the extrinsic matrix that contains the parameters of all involved external parameters and construct the extrinsic matrix.<br><br>

We can now transform the world co-ordinates to image co-ordinates with the transformations we have crafted thus far.<br><br>

![19](/assets/images/19.png){:class="img-responsive"}<br>

With this we have calibrated our camera. <br><br>

<h2>Triangulation using 2 Cameras</h2><br>
We first have to set up 2 cameras which have been calibrated and are set up apart horizontally by a known distance. There can be no difference in any other aspects of orientation other than the horizontal distance it is set apart by. The kind of stereo vision that is generated is called binocular vision. <br>
The idea is that 2 cameras are seperated by a realtively small horizontal distance. Based on this fact the images captured will be slightly translated relative to one other. So we would need to calibrate both cameras based on earlier but with a single caveat being that the camera that was shifted to the right would have to account for the shift in the horizontal direction.<br>

![20](/assets/images/20.png){:class="img-responsive"}<br>
![21](/assets/images/21.png){:class="img-responsive"}<br>

From the calibration we know of the intrinsic parameters and can re-arrange the equation to find the world co-ordinates. <br><br>
![22](/assets/images/22.png){:class="img-responsive"}<br>
![23](/assets/images/23.png){:class="img-responsive"}<br>
![24](/assets/images/24.png){:class="img-responsive"}<br>

<h2>Project</h2>

The motivation for our project is to create a platform for doing depth perception with two cheap Android phones utilizing stereo vision. There are multiple ways to perceive depth including the use of neural networks however doing it in real time with hardware has many pitfalls which we aim to uncover.

Performing depth perception with readily available hardware (such as old android phones) can be useful for hobbyists, researchers or educationists for integration into larger projects such as localization and mapping with mobile robotics.

<h2>The Setup</h2>

Our setup consisted of two android cameras being used as wireless IP cameras. <br>

{:refdef: style="text-align: center;"}
![25](/assets/images/25.png){:class="img-responsive"}{:height="300px" width="400px"}<br><br>
{: refdef}

<b>Cameras being used:</b><br><br>

![26](/assets/images/26.png){:class="img-responsive"}<br><br>


We used the application iVcam to stream the video signals from both the cell phones and use the PC client to receive it. We could have directly accessed the phones on the network via openCV however there was a significant lag of as much as ten seconds in the feed. 

It is important to note that for a robust real time system to work the video feeds must be synchronized in time, this is impossible as both cameras run on their individual system clocks.
To minimize the delay we used iVcam’s drivers to receive the video and in our code we used cv.grab and cv.retrieve to get the two frames and decode them after respectively. The alternative is to use cv.read which <i>grabs</i> and <i>retrieves</i> together therefore creating the delay of one frame being decoded. 

The above measures minimized the delay and iVcam’s client allowed us to tweak the attributes such as exposure, ISO and focus to maintain consistency. The automatic dynamics of the cameras operating individually can create significant problems as the pixel values vary due to lighting and other noises between the two images. In an ideal scenario we want two same images with a fully aligned y axis with only a difference in the x axis to produce disparity.

The best hardware setup is to have the exact same camera setup rigidly connected sharing the same system clock such as the [StereoPi](https://www.stereopi.com/).

<h2>Calibration and Rectification</h2>

The first step for creating a disparity map is to calibrate the two cameras being used and use the calibration data for rectification of both source images. The theory of camera calibration has been mentioned in detail here. The main idea with rectification is to align the images from the two cameras so that their y-axis are completely aligned. The y-axis alignment is important as when we compute disparity the search for matching points is executed along one row of pixels to keep it fast.

The first step is to determine the relationship between the cameras and their intrinsic parameters. We follow the standard way to do this by using a checkerboard and listing the matches between the two images using cv.findChessboardCorners. As this is done within openCV it is important to verify the results by using cv.drawChessboardCorners and viewing the resulting images to check if the matches were done correctly.

Successful matches should be as follows where all the number of the corners defined in the code are mapped and have the same color in the two images:

![27](/assets/images/27.png){:class="img-responsive"}<br><br>

Possible issues at this stage can be that the corners are not found at all or the points are erroneously identified. If the corners are not found there is another function cv.findChessboardCornersSB that can be used which uses a different algorithm and in our experience performed better, however ideally this should not be required.

Due to low resolution or sharp angles often the matches on the board were not the same as in the following:

![28](/assets/images/28.png){:class="img-responsive"}<br><br>

Another difficult to detect issue is when one checkerboard is detected as inverted, one can know this if the colors do not align as in the following:

![29](/assets/images/29.png){:class="img-responsive"}<br><br>

It is important to remove any such images as even one such pair of images can distort the output and create errors with the rectification.

Following the identification of the points the intrinsic and extrinsic camera matrices calculation is straight forward utilizing cv.calibrateCamera and cv.stereoCalibrate. These matrices are then used in cv.stereoRectify to get the mapping matrices, finally cv.initUndistortRectifyMap is used to create the stereo map. It is useful to use the roi_L and roi_R output of stereoRectify to understand any bugs; it outputs coordinates for a rectangle for both the images that the program considers as overlapping data. 

Finally, the stereo map can be used to rectify any pair of images from the camera setup using cv.remap. A good test for successful calibration and rectification is to overlap the two images using cv.addWeighted and check if the y axis are fully aligned as in the following image:

![30](/assets/images/30.png){:class="img-responsive"}<br><br>


A poorly overlayed example could be:

![31](/assets/images/31.png){:class="img-responsive"}<br><br>


There are multiple issues that are possible in calibration and rectification and it is important to test at the intermediate steps utilizing the methods mentioned above.

<h2>Creating the disparity map</h2>

Once images have been rectified and are aligned the disparity map can be created using the function cv.StereoBM or cv.StereooSGBM. The function has multiple parameters that directly affect the output and their respective descriptions can be read here. It can be useful to create a tool to alter the parameters in real time and see the results. We altered a code to work with static images to experiment with different settings of the parameters.

<h2>Post processing the disparity map</h2>
Post processing of the disparity map is an important step for achieving better results. In our experiment, the cv.ximgproc.createDisparityWLSFilter() is used. More on it can be read on the official [OpenCV tutorial](https://docs.opencv.org/4.x/d3/d14/tutorial_ximgproc_disparity_filtering.html) , as well as this [Stackoverflow discussion](https://stackoverflow.com/questions/62627109/how-do-you-use-opencvs-disparitywlsfilter-in-python). Our implementation uses both disparity maps - the left and right one, to create a cleaner version. The following examples show the difference between a raw disparity map and a filtered one:

{:refdef: style="text-align: center;"}
![32](assets/images/32.png){:class="img-responsive"}{:height="747px" width="381px"}<br><br>
{: refdef}


<h2>Real-time Rectification method</h2>
`An` alternative to the precalculated fundamental matrix that is achieved using the chessboard method, would be to calculate the fundamental matrix for each individual image based on the matches between the images. This method is easier to implement in comparison to the chessboard calibration, where you have to take a whole set of photos and calibrate over them. Here is the approximate pipeline one would use to achieve this:
<ol>
<li>First, we need to rectify our images, to achieve vertical alignment. Instead of using a precalculated matrix for the two cameras, we rectify the images only based on the matches between them. Thus, we need to find those matches. We need to find keypoints in both images, so we could use something like <code>sift = cv.SIFT_create()</code> and <code>kp1, des1 = sift.detectAndCompute()</code>.</li><br>
<li>Then we need to match the keypoints, using cv.FlannBasedMatcher(index_params, search_params).</li><br>
<li>What if there are multiple matches to the same keypoints? We can use RANSAC to filter for only good matches. In our implementation, RANSAC is used internally, when the fundamental matrix is being found by fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)</li><br>
<li>The fundamental matrix establishes a connection between the images, but we need to warp them somehow, to align the vertical levels of the images. We do this by _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)).</li><br>
<li>The final step is to apply the transformation to both images, this can be done by using<br>
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))<br>
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))<br></li><br>
<li>Profit! We have rectified our images, and now we can use the standard procedure to calculate the disparity.</li>
</ol>
Consider reading more on matching on the official [OpenCV tutorial](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html). We also used this [great tutorial](https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/) to understand this method.

<h2>Comparison between the Real-time and Precalculated Rectification.</h2>
Comparing these two methods, we determined a number of pros and cons for both of them:<br>
Real-time rectification Pros:
<ul>
<li>Easier to implement than Precalculated: no need to have a stereo camera setup, or take a set of chessboard photos with it. You just need two individual images of the same scene, where feature matching can happen.</li><br>
<li>Even though the results might be quite random (RANSAC is used), gives quite decent results.</li><br>
</ul>
Real-time rectification Cons:<br>
<ul>
<li>The results are not consistent. If you run the algorithm multiple times on the same pair of images, you get very different results sometimes, This is due to randomized feature matching. Each time you get a different fundamental matrix and different rectification.</li><br>
<li>The algorithm can be computationally intensive, especially if one would consider doing depth maps for a video.</li><br>
<li>The rectification transformation that is applied here can distort the image a significant amount, making the disparity map somewhat unclear.</li><br>
</ul>
Precalculated rectification Pros:<br>
<ul>
<li>The results produced are consistent throughout each image. The transformation is always the same, since the fundamental matrix is precalculated.</li><br>
<li>Faster processing times. No need to do matching!</li><br>
<li>Can be used for creating disparity maps for video. This is due to the consistency of the rectification as well as the processing times.</li>
</ul>
Precalculated rectification Cons:<br>
<ul>
<li>Complicated setup procedure, as you require a stereo camera setup. In our circumstances, rubber banding two cell phones together proved to be flimsy, so for each displacement of the cameras, ideally you have to rectify again. </li><br>
<li>Rectification using the chessboard might be buggy, you need to pay attention to results, as discussed in the “Calibration and Rectification”.  </li>
</ul>


<h2>Future Work</h2>
<ol>
<li>The next most obvious step for future work is going from the disparity map to a depth map. For this the transformation is as follows:<br><br>

Depth (mm) = baseline distance (mm) * focal length (pixels) / disparity (pixels)<br><br>

The focal length in pixels can be obtained using the following:<br><br>

F (pixels) = ImageWidth (pixel) * F (mm) / SensorWidth (mm)<br><br>

Once a depth map has been created there are multiple integrations that can be done with real time robotic systems.</li><br>
 
<li>A 3D Printed Stand can be used to hold the cameras together with markings to have reproducible results and lower noise. Evidently our setup with a clipper and rubberbands is not very rigid if we move it around.</li><br>

<li>This setup can also be used to train a neural network. LiDAR and radar are being commonly used to teach depth on a single image. A cheaper way could be to use a stereo setup to train an algorithm for research purposes.</li></ol><br>



<h2>References and other cool links</h2>
<ol>

 

<li><a href="https://www.youtube.com/watch?v=LR0bDLCElKg">Teslas work in depth perception.</a></li><br>

<li>https://giphy.com/channel/OnABottle</li></ol>

[Github](https://github.com/steiva/me455project)




[Disparity Gif](/assets/Gifs/disparity.gif)
<iframe src="https://giphy.com/embed/g4UmwJg55GpuEGoMUB" width="960" height="400" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
<iframe src="https://giphy.com/embed/zF7atpM8PXi1nMiiRH" width="480" height="200" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
<iframe src="https://giphy.com/embed/Wja0v1r3Fs23QYRGJU" width="480" height="200" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>



You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/steiva/me455project
[jekyll-talk]: https://talk.jekyllrb.com/
