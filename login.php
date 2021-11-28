<?php
$error = array('email'=>'','password'=>'');
if(isset($_POST['submit1']))
{
	$email=$_POST['email'];
	$pass=$_POST['password'];

	$servername='localhost';
	$user='root';
	 $password='';
	$db='register';
	$con=mysqli_connect($servername,$user,$password,$db);
	if(!$con)
	{
		echo "Unable to connect to database".mysqli_connect_error();
	}
	$query0="SELECT * FROM information WHERE Email='$email'";
	$result0=mysqli_fetch_assoc(mysqli_query($con,$query0));
	if($result0['Email']!=$email)
	{
		$error['email']="Not a valid user";
	}
	else{$query1="SELECT * FROM information WHERE Password='$pass'";
	$result1=mysqli_fetch_assoc(mysqli_query($con,$query1));
	if($result1['Password']!=$pass)
	{
		$error['password']="Incorrect password";
	}
}
	if(!array_filter($error))
	{
		$query2="SELECT * FROM information WHERE Email='$email' AND Password='$pass'  ";
		$result2=mysqli_fetch_assoc(mysqli_query($con,$query2));
		session_start();
		$_SESSION['username'] = $result2['Username'];
		$_SESSION['password'] = $pass;
		$_SESSION['status']="Active";
		header("location:home.php");
	}
}
?>

<!DOCTYPE html>
<html>
<head>
<title> Login Form</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<!-- Custom Theme files -->
<link href="registration.css" rel="stylesheet" type="text/css" media="all" />

</head>
<body>

	<div class="main-w3layouts wrapper">
		<h1><font color="#a5c422"><b>Login Form</b> </font></h1>
		<div class="main-agileinfo">
			<div class="agileits-top">
				<form action="#" method="post">
					<input class="text email" type="email" name="email" placeholder="Email" required="">
					<div class="red-text"><?php echo $error['email']; ?></div><br>
					<input class="text" type="password" name="password" placeholder="Password" required="">
					<div class="red-text"><?php echo $error['password']; ?></div><br>
					<div class="wthree-text">

						<div class="clear"> </div>
					</div>
					<input type="submit" name='submit1' value="LOGIN">
				</form>
				<p>Don't have an Account? <a href="registration.php"> Create account!</a></p>
			</div>
		</div>


	</div>

</body>
</html>
