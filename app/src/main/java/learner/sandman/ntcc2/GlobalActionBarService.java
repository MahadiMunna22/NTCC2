package learner.sandman.ntcc2;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.graphics.PixelFormat;
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

import com.example.android.globalactionbarservice.R;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;


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
	CascadeClassifier haarCascade;
	File mCascadeFile;

	@Override
	protected void onServiceConnected() {
		Log.d("TAG1","STARTed service");
		//**************SETTINGP UP OPENCV FOR USE****************************//
		if(OpenCVLoader.initDebug()){
			Log.d("TAG1","OpenCv started successfully");
		}

		//******************MAKING A LAYOUT FOR THE FACE*************************//
		//faceFrameLayout=new FrameLayout(this);
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
		cursorParams.gravity=Gravity.CENTER;
		Log.d("TAG1","I CAME HERE");
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
			bringInTheCascadeFile();
		} catch (IOException e) {
			e.printStackTrace();
		}
		//the function below loads the haarcascade classifier variable from the concerned files
		openTheBroughtInCascadeFile();


		//********************GETTING MY SCREEN MEASUREMENTS************************//
		DisplayMetrics displayMetrics = new DisplayMetrics();
		myWindowManager.getDefaultDisplay().getMetrics(displayMetrics);
		final int height = displayMetrics.heightPixels;
		final int width = displayMetrics.widthPixels;

		//*********MAKING THE HANDLER TO RECIEVE THE CO_ORDS*********************//



		handler=new Handler(){
			int oldX=0,oldY=0;
			@Override
			public void handleMessage(Message msg) {

				int cursorValueX=width/(120)*msg.arg1-1000;
				int cursorValueY=height/(50)*msg.arg2-2300;
				/*if((Math.abs(cursorValueX-oldX)>25) || (Math.abs(cursorValueY-oldY)>25)  ){//if signigficant change
					//we let the cursor chage and also store value
					oldX=cursorValueX;oldY=cursorValueY;
				}else{
					//we again use the old value
					cursorValueX=oldX;
					cursorValueY=oldY;
				}*/
				//we update the cursor based on new parameters
				cursorParams.x=cursorValueX;
				cursorParams.y=cursorValueY;
				Log.d("TAG1","face :x="+cursorValueX+",y="+cursorValueY);
				myWindowManager.updateViewLayout(cursorFrameLayout,cursorParams);


			}
		};


	}

	//variables related to opencv
	//this var is going to keep the count of all frmes processed from the start
	int frameCount=0;

	//*************************FUNCTIONS RELATED TO OPENCV****************************//
	@Override
	public void onCameraViewStarted(int width, int height) {

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
		Imgproc.resize(mRgbat,mRgbat,mRgba.size());
		//setting up the greyscale matrix
		Mat mGray=inputFrame.gray();
		Mat mGrayt=mGray.t();
		Core.flip(mGrayt,mGrayt,-1);
		Imgproc.resize(mGrayt,mGrayt,mGray.size());
		//doing the detection work
		MatOfRect faces = new MatOfRect();

		//********************************PART 1*********************************************//
		if(haarCascade != null) {
			//Log.d("TAG1","Detection going on");
			haarCascade.detectMultiScale(mGrayt, faces, 1.1, 2,
					2, new Size(100,100), new Size());

		}
		//*****************************************************************************//
		//*******************************PART 2**********************************************//
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++) {
			Imgproc.rectangle(mRgbat, facesArray[i].tl(),facesArray[i].br(), new Scalar(100), 3);
			Message message=Message.obtain();
			message.arg1=facesArray[i].x;
			message.arg2=facesArray[i].y;
			handler.sendMessage(message);
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			//facesArray[i].x
		}



		return mRgbat;
	}


	void bringInTheCascadeFile() throws IOException {
		InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_default);//smile er jonno cascade file ta nilam
		File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);//directory banalam jeta private
		mCascadeFile = new File(cascadeDir,"cascade.xml");//directoryr moddhe cascade.xml file ta rakhlam
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
	}

	void openTheBroughtInCascadeFile(){
		haarCascade= new CascadeClassifier(mCascadeFile.getAbsolutePath());//ekta cascade classifier banalam using the file
		if(!haarCascade.empty()){
			Log.d("TAG1","The haar Cascde object ain't empty");
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
