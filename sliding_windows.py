from keras.applications.resnet50 import ResNet50
from keras.models import Model
import cv2
from keras.optimizers import Adam, Nadam
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
import math
import tensorflow as tf


def mergeRectangles(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	return boxes[pick]

def sliding_window(src_video):
    
    cap = cv2.VideoCapture(src_video)
    current_fram_number = 0

    while(cap.isOpened()):

      ret, frame = cap.read()
      if ret == False:
        break

      scales = [0.5, 0.75, 1, 1.25, 1.5] # scale the entire image to get different "window sizes"
      boxes = []
      predictions = []
      
      # collect predictions per each frame at each scale
      for scale in scales:
        windows = []
        x_coords = []
        y_coords = []
        im_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmp = frame
        count = 0
        new_dim = (int(im_frame.shape[1]*scale), int(im_frame.shape[0]*scale))
        im_frame = cv2.resize(im_frame, new_dim, interpolation = cv2.INTER_AREA)

        for x in range(0, im_frame.shape[1] - w_width, stepSize):
            for y in range(0, im_frame.shape[0] - w_height, stepSize):
                window = im_frame[x:x + w_width, y:y + w_height]

                if window.shape[0] != 0 and window.shape[1] != 0:
                    
                    window = cv2.resize(window, (40, 40))
                    cv2.imwrite('windows/window'+str(count)+'.jpg',window)
                    window = np.reshape(window, [1, 40, 40, 1])
                    window = window / 255.
                    windows.append(window)
                    x_coords.append(x)
                    y_coords.append(y)
                    boxes.append([y / scale, x / scale, (y + w_height) / scale, (x + w_width) / scale])
                    count += 1

        windows = np.vstack(windows)
        predictions.append(model.predict(windows, steps = 1))

      predictions = np.array([i for pred in predictions for i in pred])
       
      sess = tf.compat.v1.Session()

      # NON MAXIMUM SUPPRESION
      selected_indices = tf.image.non_max_suppression(boxes = boxes, scores =  np.reshape(predictions, [predictions.shape[0],]), iou_threshold = 0.5, max_output_size = 5) 
      selected_boxes = tf.gather(boxes, selected_indices)

      dim = sess.run(tf.shape(selected_boxes))
      sb = sess.run(selected_boxes)
      
      print("Frame", current_fram_number)

      # merge results
      mergedRectangles = mergeRectangles(sb, 0)
      # then draw results on current frame
      with open("video_49_annotations.txt", "a") as file:
        for rect in mergedRectangles:
            file.write("{},{},{},{} ".format(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
            cv2.rectangle(tmp, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255, 153, 51), 2)
        file.write("\n")
        
      cv2.imwrite('video_49/' + str(current_fram_number) +'.png', tmp)
        
      current_fram_number += 1       
      cv2.imshow('Frame', tmp)
      cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":

  num_classes = 1
  img_height, img_width = 40, 40
  lr = 0.0001

  # MODEL
  base_model = VGG19(weights=None, include_top=False, input_shape=(img_height, img_width, 1))
  x = base_model.output
  x = Flatten()(x)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.7)(x)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)  
  predictions = Dense(num_classes, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  adam = Nadam(lr=lr)
  model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
  model.load_weights('weights/vgg19_binary_all_balanced.h5')

  stepSize = 20
  (w_width, w_height) = (40, 40)  # window size
  
  src_video = 'originals/Video_49.avi'
  sliding_window(src_video)
