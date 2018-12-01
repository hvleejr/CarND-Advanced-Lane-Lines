
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Undistorted"
[image2]: ./output_images/undistort_actual.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/extracted.png "Output"
[image7]: ./output_images/pipeline.png "Pipeline"
[image72]: ./output_images/pipeline2.png "Pipeline"
[video1]: ./project_video.mp4 "Video"
[image8]: ./output_images/fail_lines.png "Fail Lines"
[image9]: ./output_images/fail_curves.png "Fail Lines"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in **cell #2** of the IPython notebook located in "project_notebook.ipynb". The camera calibration is implemented as a class `CameraCalibration`. The attributes of this class are the camera calibration constants while the methods are the calibration and warping functions. Instantiating this class requires one to provide a directory containing a series of chessboard images of varying orientations, as is the typical way to calibrate a camera.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

On the left is the distorted image while on the right is the undistorted output. The function `cv2.undistort` is used along with the constants obtained during the camera calibration step. The difference is most obvious if we look closely at the hood of the vehicle.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The thesholding function is defined in **cell #9** in "project_notebook.ipynb". The S-channel is thresholded between the interval [170, 255] while the threshold values for the Sobel transformed image is the interval [20, 100]. Both of these intervals are taken from the lecture notes.

Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

**Cells #6 and #7** of "project_notebook.ipynb" show how the perspective transform is carried out. The code for perspective transform is handled by the `CameraCalibration` class under the method `warp_image()`. The method `calc_warp_matrices()` takes in two sets of points `src` and `dst` and calculates for both the forward and reverse warp matrices. These matrices are then stored as class attributes `M` and `Minv`. The `src` and `dst` points are defined in the code:

``` python
# Warp constants
src = np.float32([[574, 462], # top left
                  [243, 684], # bottom left
                  [1056, 679], # bottom right
                  [704, 462]]) # top right
offset = 250
dst = np.float32([[offset, 100],      
                  [offset, CC.H],
                  [CC.W-offset, CC.H],
                  [CC.W-offset, 100]])
```

The variables `CC.H` and `CC.W` are the image height and widths respectively. 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 574, 462      | 250, 100        | 
| 243, 684      | 250, 720      |
| 1056, 679     | 1030, 720      |
| 704, 462      | 1030, 100        |

Below is an example of the image before and after warping. The `src` and `dst` points are outlined by the red polygons on both images. In this example, an image of a straight lane produces a pair of parallel lines when transformed.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

From the warped binary image, the pixels for the left and right lines are extracted by sliding a window along each of the lines, starting from the bottom going to the top of the image. As the window moves up, its position is updated by shifting the center to mean location of the white pixels. This whole procedure is handled by the function `extract_lanes()` in **cell #11**. 

Additionally, a class `LaneLine` is defined, whose attributes are the line positions and polynomial fit parameters. The class methods are `compute_fit()` and `compute_R()` which computes for the polynomial fit and radius of curvature respectively. 

To speed up the process of pixel extraction, the position of the previous lines can be used for estimating the search area for the next line positions. This is handled by the code:
```python
# search around previous line 
# ..left line
left_lane_inds = ((left.fit[0]*nonzeroy**2 + 
                   left.fit[1]*nonzeroy + 
                   left.fit[2] - margin < nonzerox) & 
                  (left.fit[0]*nonzeroy**2 + 
                   left.fit[1]*nonzeroy + 
                   left.fit[2] + margin > nonzerox)).nonzero()[0]
# ..right line
right_lane_inds = ((right.fit[0]*nonzeroy**2 + 
                    right.fit[1]*nonzeroy + 
                    right.fit[2] - margin < nonzerox) &
                    (right.fit[0]*nonzeroy**2 + 
                     right.fit[1]*nonzeroy + 
                     right.fit[2] + margin > nonzerox)).nonzero()[0]

```
In the code above, `left` and `right` are objects corresponding to the left and right lines respectively. After extracting the line pixels, a 2nd degree polynomial is fitted over the pixel points like in the image below. 

![alt text][image5]

The fitting is done through the `compute_fit()` method of th `left` and `right` objects. The numpy function `polyfit()` is used in this process.
```python
    def compute_fit(self, deg=2):
        self.fit = np.polyfit(self.y, self.x, deg)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After getting the polynomial fits, the radius of curvature is calculated by the class method:
```python
    def compute_R(self, xm_per_pix, ym_per_pix, y_eval):
        '''
        xm_per_pix = meters per pix in x
        ym_per_pix = meters per pix in y
        y_eval = y position in meters
        '''
        fit_cr = np.polyfit(self.y * ym_per_pix, self.x * xm_per_pix, 2)
        self.R = ((1+(2*fit_cr[0]*y_eval + 
                  fit_cr[1])**2)**1.5) / (np.abs(2*fit_cr[0]))
```

The conversion factors are hardcoded as:
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/np.mean(right_fitx - left_fitx) # average pixel distance between lines
```
Finally, the deviation of the car from the lane center is calculated first by getting the distance of the car from each of the lines:

```python
# Compute for distance from lane to car (in meters)
midpoint = out_img.shape[1] // 2
left.dist_to_camera = (midpoint - left_fitx[-1]) * xm_per_pix
right.dist_to_camera = (right_fitx[-1] - midpoint) * xm_per_pix
```

Then by subtracting the distance of the car from the left lane with the distance of the car from the right lane. Here the calculation for the deviation is done inside the `annotate()` function, which handles the annotation of each frame.

```python
def annotate(img, left, right):
    thickness = 2
    scale = 1.5
    color = (255,255,255)
    cv2.putText(img, 
            'radius: [{:.2f}m, {:.2f}m]'.format(left.R, right.R),
            (10,100), cv2.FONT_HERSHEY_DUPLEX, 
            scale, color, thickness)
    deviation = left.dist_to_camera - right.dist_to_camera
    cv2.putText(img, 
            'deviation: {:.2f}m'.format(deviation),
            (10,150), cv2.FONT_HERSHEY_DUPLEX, 
            scale, color, thickness)
    return img
```

All in all, the line extraction step can be visualized by the image below. The green boxes represent the sliding window positions during pixel extraction. The red and blue pixels correspond to the extracted pixels for the left and right lines respectively. The blue and orange lines are the fitted lines whose radii of curvature are written on top of the image.

![alt text][image6]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In **cell #13** of "project_notebook.ipynb", the function `generate_overlay()` creates an image showing the following features: the highlighted lane pixels, the shaded lane region, and the fitted polynomial lines. This overlay is merged with the original image using the `merge_overlay()` function in order to create the desired final output. This function performs an inverse perspective transform by using the `cv2.warpPerspective()` function with the inverse matrix `Minv` as input parameter. The entire pipeline is summarized in the series of images below.

![alt text][image7]
![alt text][image72]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The two challenge videos `challenge_video.mp4` and `harder_challenge_video.mp4` show scenarios where the current pipeline would likely fail. The file "project_notebook-challenges.ipynb" shows how the pipeline fails in these instances. 

First is the issue of falsely detecting lines on the road. This may arise from lines in the asphalt or from barriers. An example is shown in the image below where the pipeline incorrectly identified the road seam as a lane line. Furthermore, the binarization failed to get a clear outline of the yellow lane on the left. This can be possibly fixed by modifying the binarization procedure. For instance, one can define a finite color gamut for the road markings (e.g. yellow and white). One can also modify the way color thresholding and edge detection is combined such that false lines won't appear so strong in the binary image. 

![alt text][image8]

The next issue is when the road curves sharply such that the region of interest becomes inaccurate. An example of this is shown below. This leads to two problems - the lines are no longer quadratic, and a lot of undesired features gets included in the scene. A possible solution to this problem would be to implement an adaptive region of interest which follows the direction of the road.

![alt text][image9]
