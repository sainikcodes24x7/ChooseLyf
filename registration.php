<?php
$username=$mypassword=$email='';
$error = array('confirm_password'=>'','message'=>'','username'=>'','password'=>'');
if(isset($_POST['submit']))
{
$mypassword =$_POST['password'];
if($_POST['confirm_password']!=$mypassword)
    $error['confirm_password']='Please confirm your password';
    $servername='localhost';
    $user='root';
     $password='';
    $db='register';
    $username =$_POST['username'];
    $email =$_POST['email'];
    if(strlen($mypassword)<=6)
    $error['password']="Weak password";
    $con=mysqli_connect($servername,$user,$password,$db);
    if(!$con)
    {
      echo "Unable to connect to database".mysqli_connect_error();
    }
    $query0="SELECT * FROM information WHERE Email='$email' AND Password='$mypassword'";
    $result0=mysqli_query($con,$query0);
    $res0=mysqli_fetch_assoc($result0);
    if($res0['Email']==$email && $res0['Password']==$mypassword)
    {
      $error['message']="Already registered please login";
    }
    $query1="SELECT * FROM information WHERE Username='$username'";
    $result1=mysqli_fetch_assoc(mysqli_query($con,$query1));
    if($result1['Username']==$username)
    {
      $error['username']="Username already taken please enter another";
    }
if(!array_filter($error))
  {
    $query="INSERT INTO information(Username,Email,Password) VALUES ('$username','$email','$mypassword')";
    $result=mysqli_query($con,$query);
    if(!$result)
    {
      echo "Query not working";
    }
    if($result)
    {
      session_start();
      $_SESSION['username'] = $username;
      $_SESSION['password'] = $mypasswordpass;
      $_SESSION['status']="Active";
      header("location:home.php");
    }
  }
}
?>
<!DOCTYPE html>
<html>
<head>
<title>Registration Form</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />

<link href="registration.css" rel="stylesheet" type="text/css" media="all" />

</head>
<body>

	<div class="main-w3layouts wrapper">
    <div class="classfont">
		<h1><font color="#a5c422"><b>Registration Form</b> </font></h1>
    </div>
		<div class="main-agileinfo">
			<div class="agileits-top">
				<form action="#" method="post">
          <div class="red-text"><?php echo $error['message']; ?></div><br>
					<input class="text" type="text" name="username" placeholder="Username" required="">
          <div class="red-text"><?php echo $error['username']; ?></div><br>
					<input class="text email" type="email" name="email" placeholder="Email" required="">
					<input class="text" type="password" name="password" placeholder="Password" required="">
          <div class="red-text"><?php echo $error['password']; ?></div><br>
					<input class="text w3lpass" type="password" name="confirm_password" placeholder="Confirm Password" required="">
          <div class="red-text"><?php echo $error['confirm_password']; ?></div><br>
					<div class="wthree-text">
						<label class="anim">
							<input type="checkbox" class="checkbox" required="">
							<span>I Agree To The Terms & Conditions</span>
						</label>
						<div class="clear"> </div>
					</div>
					<input type="submit" name='submit' value="SIGNUP">
				</form>
				<p>Already have an Account? <a href="login.php"> Login Now!</a></p>
			</div>
		</div>


	</div>

</body>
</html>
