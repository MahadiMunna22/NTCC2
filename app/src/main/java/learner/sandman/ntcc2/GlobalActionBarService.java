package learner.sandman.ntcc2;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.widget.FrameLayout;
import android.widget.Toast;

import com.example.android.globalactionbarservice.R;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;


public class GlobalActionBarService extends AccessibilityService implements CameraBridgeViewBase.CvCameraViewListener2 {
	//handler
	Handler handler;
	//variables used for windowmanagement of faceView
	WindowManager myWindowManager;
	WindowManager.LayoutParams faceParams;
	View faceView;
	//variables used for the windowmanagement of the cursor
	FrameLayout cursorFrameLayout;
	WindowManager.LayoutParams cursorParams;
	View cursorView;
	//variables related to cameraview and camera listeners
	CameraBridgeViewBase cameraView;
	//files and classifiers for image processing
	CascadeClassifier haarCascadeClassifierForFace, haarCascadeClassifierForTeeth;


	//making it global for some specific purposes

	Rect boxForKLT;

	MatOfPoint features;
	Mat mPrevGrayt;
	MatOfPoint2f prevFeatures,nextFeatures;
	MatOfByte status;
	MatOfFloat err;
	Point nosePoint,eyePoint1,eyePoint2;

	@Override
	protected void onServiceConnected() {
		//**************SETTINGP UP OPENCV FOR USE****************************//
		if(OpenCVLoader.initDebug()){
			Log.d("TAG1","OpenCv started successfully");
		}
		//******************MAKING A LAYOUT FOR THE FACE*************************//
		//getting ref to the faceLayout
		faceView= LayoutInflater.from(this).inflate(R.layout.face_layout,null);
		//getting a windowmanager
		myWindowManager= (WindowManager) getSystemService(WINDOW_SERVICE);
		//setting up the parameters for my layout
		faceParams= new WindowManager.LayoutParams(
				WindowManager.LayoutParams.WRAP_CONTENT,
				WindowManager.LayoutParams.WRAP_CONTENT,
				WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
				WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE| WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
				PixelFormat.TRANSLUCENT
		);
		faceParams.gravity= Gravity.TOP|Gravity.LEFT;
		faceParams.x=0;
		faceParams.y=0;
		//faceParams.alpha= (float) 0.005;
		//telling windowmanager to add my faceview on  screen top using my params
		myWindowManager.addView(faceView,faceParams);
		//**********************MAKING A LAYOUT FOR THE CURSOR************************//
		cursorFrameLayout=new FrameLayout(this);
		cursorParams=new WindowManager.LayoutParams(
				WindowManager.LayoutParams.WRAP_CONTENT,
				WindowManager.LayoutParams.WRAP_CONTENT,
				WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
				WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE|WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
				PixelFormat.TRANSLUCENT
		);
		cursorParams.gravity=Gravity.TOP|Gravity.LEFT;
		LayoutInflater inflater=LayoutInflater.from(this);
		Log.d("TAG1","I CAME HERE2");
		inflater.inflate(R.layout.cursor_layout,cursorFrameLayout);
		Log.d("TAG1","I CAME HERE3");
		//use cursorFramlayout instead of cursor view
		myWindowManager.addView(cursorFrameLayout,cursorParams);
		//***********************SETTING UP THE FACEVIEWS AND CAMERALISTENERS***********************//
		cameraView=faceView.findViewById(R.id.cameraView);
		cameraView.setVisibility(SurfaceView.VISIBLE);
		cameraView.setCvCameraViewListener(this);
		cameraView.enableView();
		//********************OPENING CASCADE FILES AND SETTING UP THE CLASSIFIERS
		try {
			bringInTheCascadeFileForFaceDetection();
			bringInTheCascadeFileForTeethDetection();
		} catch (IOException e) {
			e.printStackTrace();
		}
		//********************GETTING MY SCREEN MEASUREMENTS************************//
		DisplayMetrics displayMetrics = new DisplayMetrics();
		myWindowManager.getDefaultDisplay().getMetrics(displayMetrics);
		final int screenHeight = displayMetrics.heightPixels;
		final int screenWidth = displayMetrics.widthPixels;

		Log.d("TAG1","screenHeight="+screenHeight+",screenWidth="+screenWidth);

		//*********MAKING THE HANDLER TO RECIEVE THE CO_ORDS*********************//



		handler=new Handler(){

			int centralXPositiononBox,centralYPositiononBox;

			int oldXPositionOnBox, oldYPositionOnBox;
			int oldXpositionOnScreen,oldYPositionOnScreen;

			int xPositionOnBox,yPositionOnBox;
			int xBoxWidth,yBoxHeight;

			int xPositionOnScreen,yPositionOnScreen;
			int xScreenWidth=screenWidth*1,yScreenHeight=screenHeight*1;


			int violaJonesUsed=0;


			@Override
			public void handleMessage(Message msg) {
				//handleCursorMovementUsingTheMessage(msg);
				handleCursorMovementUsingTheMessage2(msg);

			}

			private void handleCursorMovementUsingTheMessage2(Message msg){
				//collecting the bundle from the message as received Bundle
				Bundle receivedBundle=msg.getData();
				//making an intefer array of length 4 to store
				//1)x co-ordinate 2)y co-ordinate 3)row size 4)col-size
				int[] receivedValues=receivedBundle.getIntArray("message");
				xPositionOnBox=receivedValues[0];
				yPositionOnBox=receivedValues[1];
				xBoxWidth=receivedValues[2];
				yBoxHeight=receivedValues[3];
				violaJonesUsed=receivedValues[4];
				//Log.d("TAG2","I came here");

				//if i have not gotten the first message and this is the very first message
				if(violaJonesUsed ==1){
					//i declare that i have received the very first message
					violaJonesUsed =0;
					centralXPositiononBox= (int) nosePoint.x;
					centralYPositiononBox= (int) nosePoint.y;

					xPositionOnScreen=(xScreenWidth/xBoxWidth)*xPositionOnBox;
					yPositionOnScreen=(yScreenHeight/yBoxHeight)*yPositionOnBox;

					cursorParams.x=xPositionOnScreen;
					cursorParams.y=yPositionOnScreen;
					//Log.d("TAG2","cursorParams.x="+cursorParams.x+",cursorParams.y="+cursorParams.y);
					myWindowManager.updateViewLayout(cursorFrameLayout,cursorParams);

					oldXPositionOnBox=xPositionOnBox;
					oldYPositionOnBox=yPositionOnBox;
					oldXpositionOnScreen=xPositionOnScreen;
					oldYPositionOnScreen=yPositionOnScreen;

				}else{
					//if i moved right in box
					if(xPositionOnBox-centralXPositiononBox>20){
						//the cursor moves right by 20 pixels
						xPositionOnScreen=oldXpositionOnScreen+25;
					}//if i moved left
					else if(xPositionOnBox-centralXPositiononBox<-20){
						xPositionOnScreen=oldXpositionOnScreen-25;

					}

					//if i moved down in the box
					if(yPositionOnBox-centralYPositiononBox>20){
						//the cursor moves right by 20 pixels
						yPositionOnScreen=oldYPositionOnScreen+25;
					}//if i moved left
					else if(yPositionOnBox-centralYPositiononBox<-20){
						yPositionOnScreen=oldYPositionOnScreen-25;

					}


					oldXPositionOnBox=xPositionOnBox;
					oldYPositionOnBox=yPositionOnBox;
					oldXpositionOnScreen=xPositionOnScreen;
					oldYPositionOnScreen=yPositionOnScreen;

					cursorParams.x=xPositionOnScreen;
					cursorParams.y=yPositionOnScreen;
					Log.d("TAG2","cursorParams.x="+cursorParams.x+",cursorParams.y="+cursorParams.y);
					myWindowManager.updateViewLayout(cursorFrameLayout,cursorParams);


				}

			}


		};


	}
	//variables related to opencv
	//this var is going to keep the count of all frmes processed from the start
	int frameCount=0;

	//*************************FUNCTIONS RELATED TO OPENCV****************************//
	@Override
	public void onCameraViewStarted(int width, int height) {
		//these variables are used in KLT
		mPrevGrayt = new Mat();
		features = new MatOfPoint();
		prevFeatures = new MatOfPoint2f();
		nextFeatures = new MatOfPoint2f();
		status = new MatOfByte();
		err = new MatOfFloat();
		nosePoint=new Point();
	}

	@Override
	public void onCameraViewStopped() {

	}

	@Override
	public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
		//increasing frame counts
		frameCount++;
		//if the  total number of frames processed till now becomes 10k, then we reset the counter
		if(frameCount==10000){
			frameCount=0;
		}
		//setting up the RGBA matrix
		Mat mRgba=inputFrame.rgba();
		Mat mRgbat=mRgba.t();
		Core.flip(mRgbat,mRgbat,-1);
		//Imgproc.resize(mRgbat,mRgbat,mRgba.size());
		Imgproc.resize(mRgbat,mRgbat,new Size(mRgba.width(),mRgba.height()),2,1,Imgproc.INTER_LINEAR);
		//setting up the greyscale matrix
		Mat mGray=inputFrame.gray();
		Mat mGrayt=mGray.t();
		Core.flip(mGrayt,mGrayt,-1);
		//Imgproc.resize(mGrayt,mGrayt,mGray.size());
		Imgproc.resize(mGrayt,mGrayt,new Size(mGray.width(),mGray.height()),2,1,Imgproc.INTER_LINEAR);
		//doing the detection work

		runViolaJonesForFaceDetection(mRgbat, mGrayt);


		runViolaJonesForTeethDetection(mRgbat.submat((int) nosePoint.y,mRgbat.rows(),mRgbat.cols()*1/4,mRgbat.cols()*3/4),
													mGrayt.submat((int) nosePoint.y,mGrayt.rows(),mGrayt.cols()*1/4,mGrayt.cols()*3/4));



	runOpticalFlow(mRgbat,mGrayt);
	/*	if(boxForKLT!=null){
			runOpticalFlow(mRgbat.submat( (int)boxForKLT.tl().y,(int)boxForKLT.br().y,(int)boxForKLT.tl().x,(int)boxForKLT.tl().x),
					mGrayt.submat( (int)boxForKLT.tl().y,(int)boxForKLT.br().y,(int)boxForKLT.tl().x,(int)boxForKLT.tl().x)

			);
		}else{
			runOpticalFlow(mRgbat, mGrayt);
		}*/

		return mRgbat;

	}
	private void runViolaJonesForFaceDetection(Mat mRgbat, Mat mGrayt) {
		MatOfRect faces = new MatOfRect();
		//after every 30 frames i am using viola jones and getting the nasal co-ordinates
		if(frameCount%30==0){
			//********************************PART 1*********************************************//
			//if the classifiers are available then i do the detection and store the detected faces
			//in the array
			if(haarCascadeClassifierForFace != null) {
				haarCascadeClassifierForFace.detectMultiScale(mGrayt, faces, 1.1, 2,
						2, new Size(100,100), new Size());
			}
			//*******************************PART 2**********************************************//
			//faces array stores the detected faces...simply speaking
			Rect[]facesArray = faces.toArray();
			if(facesArray.length!=0){
				boxForKLT=facesArray[0];
			}
			//boxForKLT=facesArray[0];
			for (int i = 0; i < facesArray.length; i++) {
				//this code inserts the squares where the faces have been found
				Imgproc.rectangle(mRgbat, facesArray[i].tl(),facesArray[i].br(), new Scalar(100), 3);

				//centre point is actually the centre of the square around the face
				Point centrePoint=new Point();

				centrePoint.x=	facesArray[i].tl().x/2	+facesArray[i].br().x/2;
				centrePoint.y=	facesArray[i].tl().y/2	+facesArray[i].br().y/2;
				//i keep the centrePoint in nose point for KLT ,nosePoint is a global variable
				nosePoint=centrePoint;
				//this is done to make the feature empty so that the new viola jones
				//co-ordinates can be give the new nosePoint to the optical flow part
				features=new MatOfPoint();
				//making the this thread sleep for 10ms
				//i am doing this to let the processor core rest a bit
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}

			}

		}//this bracket is where the if for checking 20 frame count ends
	}

	private void runViolaJonesForTeethDetection(Mat mRgbat, Mat mGrayt) {
		MatOfRect teeth = new MatOfRect();

		//********************************PART 1*********************************************//
		//if the classifiers are available then i do the detection and store the detected teeth
		//in the array
		if(haarCascadeClassifierForTeeth != null) {
			haarCascadeClassifierForTeeth.detectMultiScale(mGrayt, teeth, 1.1, 4,
					2, new Size(), new Size());

		}
		//*******************************PART 2**********************************************//
		//teeth array stores the detected teeth...simply speaking
		Rect[] teethArray = teeth.toArray();
		if(teethArray.length>0)
			Log.d("TAGteeth","Teeth found ="+teethArray.length);
		for (int i = 0; i < teethArray.length; i++) {
			//this code inserts the squares where the teeth have been found
			Imgproc.rectangle(mRgbat, teethArray[i].tl(),teethArray[i].br(), new Scalar(100), 3);


			Point centrePoint=new Point();
			centrePoint.x=	teethArray[i].tl().x/2	+teethArray[i].br().x/2;
			centrePoint.y=	teethArray[i].tl().y/2	+teethArray[i].br().y/2;
			//i keep the centrePoint in nose point for KLT ,nosePoint is a global variable
			//nosePoint=centrePoint;
			//this is done to make the feature empty so that the new viola jones
			//co-ordinates can be give the new nosePoint to the optical flow part
			//features=new MatOfPoint();
			//making the this thread sleep for 10ms
			//i am doing this to let the processor core rest a bit
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

		}



	}

	private void runOpticalFlow(Mat mRgbat, Mat mGrayt) {
		//******************THIS IS THE CODE FOR OPTICAL FLOW************************************//
		//if there are no features (points available for tracking)
		if(features.toArray().length==0){
			//store this grey matrix as prev Gray Matrix
			mPrevGrayt=mGrayt.clone();
			//make a point array with only one point (for the nose co-ordinate)
			Point[]points=new Point[1];
			//i am putting in the nose co-oridnate found by viola jones algo
			points[0]=nosePoint;



			//now the features object has the nose feature a.k.a the nose
			// co-ordinates after the following code is exeuted
			features.fromArray(points);
			//i am storing these features as  prevFeatures
			prevFeatures.fromArray(features.toArray());
		}else{
			//the else part is called when there are previous features availabe
			//for comparison
			//get where the present nose point feature is and store it in nextFeatures variable
			Video.calcOpticalFlowPyrLK(mPrevGrayt, mGrayt,prevFeatures, nextFeatures, status, err);
			//we make a list of point to store the nextFeatures point
			//which is indeed just one point
			List<Point> drawFeature = nextFeatures.toList();
			Log.d("TAG1","Draw features .size="+drawFeature.size());
			//we loop on this new point(it is being redundant as there is only one point)
			for(int j = 0; j<  drawFeature.size(); j++){
				//this code just makes the mesg obj
				Message message=Message.obtain();
				//we keep the nose feauture point in point p
				//it was kept tracked by optical flow
				Point p = drawFeature.get(j);
				//we draw the point on the image
				Imgproc.circle(mRgbat, p, 5, new Scalar(255),10);
				//i am making a bundle for inserting the points
				Bundle bundle=new Bundle();
				//i am giving the tag message to the bundle contents
				//i am providing the centre co-orinates
				//i am putting the number of rows and cols i.e width and height of the matrix

				int violaJonesUsed;
				if(frameCount%30==0){
					violaJonesUsed=1;
				}else{
					violaJonesUsed=0;
				}


				bundle.putIntArray("message",

						new int[]{(int) drawFeature.get(j).x,
								(int) drawFeature.get(j).y,
								mRgbat.cols(),//width
								mRgbat.rows(), //height
								violaJonesUsed
						});

				// i am inserting this data into the bundle
				message.setData(bundle);
				//sending the message with points to the UI thread using handler to control the cursor
				handler.sendMessage(message);
			}
			//keeping this matrix as prevmatrix (the gray matrix)
			mPrevGrayt = mGrayt.clone();
			//keeping present features stored for the future as prevFeatures
			prevFeatures.fromList(nextFeatures.toList());
		}
	}

	void bringInTheCascadeFileForFaceDetection() throws IOException {
		InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);//smile er jonno cascade file ta nilam
		File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);//directory banalam jeta private
		File mCascadeFile = new File(cascadeDir,"cascade.xml");//directoryr moddhe cascade.xml file ta rakhlam
		FileOutputStream os = new FileOutputStream(mCascadeFile);
		//it is stream to write to my new cascade.xml
		//file


		byte[] buffer = new byte[4096];//ekta byte array banalam buffer naame
		int bytesRead;//this will collect a  byte of data from input stream

		while((bytesRead = is.read(buffer)) != -1)//is.read reads  from file and puts in buffer and returns koy byte porlo
		{
			os.write(buffer, 0, bytesRead);//buffer theke data niye write korche
		}
		is.close();
		os.close();
		haarCascadeClassifierForFace = new CascadeClassifier(mCascadeFile.getAbsolutePath());//ekta cascade classifier banalam using the file
		if(!haarCascadeClassifierForFace.empty()){
			Log.d("TAG1","The haar Cascde object ain't empty");
		}
	}

	void bringInTheCascadeFileForTeethDetection() throws IOException {
		InputStream is = getResources().openRawResource(R.raw.cascadefina);//smile er jonno cascade file ta nilam
		File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);//directory banalam jeta private
		File mCascadeFile = new File(cascadeDir,"cascade.xml");//directoryr moddhe cascade.xml file ta rakhlam
		FileOutputStream os = new FileOutputStream(mCascadeFile);
		//it is stream to write to my new cascade.xml
		//file


		byte[] buffer = new byte[4096];//ekta byte array banalam buffer naame
		int bytesRead;//this will collect a  byte of data from input stream

		while((bytesRead = is.read(buffer)) != -1)//is.read reads  from file and puts in buffer and returns koy byte porlo
		{
			os.write(buffer, 0, bytesRead);//buffer theke data niye write korche
		}
		is.close();
		os.close();
		haarCascadeClassifierForTeeth = new CascadeClassifier(mCascadeFile.getAbsolutePath());//ekta cascade classifier banalam using the file
		if(!haarCascadeClassifierForTeeth.empty()){
			Log.d("TAG2","The haar Cascde object for eyes ain't empty");
		}
	}



	//****************************THESE ARE NOT NEEDED************************************8
	@Override
	public void onAccessibilityEvent(AccessibilityEvent event) {
		Log.d("TAG1","STARTed service");
	}

	@Override
	public void onInterrupt() {

	}


}
