<?php
session_start();
if(isset($_POST['logout']))
{include("destroy.php");}
if($_SESSION['status']!="Active")
 {header("location:login.php");}
 if(isset($_POST['submit3'])){
 $servername='localhost';
 $username='root';
 $password='';
 $db='register';
 $name =$_POST['name'];
 $gender =$_POST['gender'];
 $age =(int)$_POST['age'];
 $address =$_POST['address'];
 $city =$_POST['city'];
 $disease =$_POST['disease'];
 $blood_group =$_POST['blood_group'];
 $contact=$_POST['contact'];
 $con=mysqli_connect($servername,$username,$password,$db);
 if(!$con)
 	{
 		echo "Unable to connect to database".mysqli_connect_error();
 	}
 $query="INSERT INTO donar(Name,Gender,Age,Address,City,Disease,Bloodgroup,Contact) VALUES ('$name','$gender',$age,'$address','$city','$disease','$blood_group','$contact')";
 $result=mysqli_query($con,$query);
 header("location:home.php");
}
?>
<!DOCTYPE html>
<html>
<head>
  <script src="https://kit.fontawesome.com/b62e73b8ee.js" crossorigin="anonymous"></script>
<title>Blood Donation Form</title>

     <meta charset="UTF-8">
     <meta http-equiv="X-UA-Compatible" content="IE=Edge">
     <meta name="description" content="">
     <meta name="keywords" content="">
     <meta name="author" content="Tooplate">
     <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">

     <link rel="stylesheet" href="css/bootstrap.min.css">
     <link rel="stylesheet" href="css/font-awesome.min.css">
     <link rel="stylesheet" href="css/animate.css">
     <link rel="stylesheet" href="css/owl.carousel.css">
     <link rel="stylesheet" href="css/owl.theme.default.min.css">
     <link href="registration.css" rel="stylesheet" type="text/css" media="all" />
     <!-- MAIN CSS -->
     <link rel="stylesheet" href="css/tooplate-style.css">
    <style>
        .searchopt{
            margin-top: 100px;
            margin-left: 30%;
        }
        .btn1{
            padding-left: 15px;
            padding-right: 15px;
            background: #a5c422;
            color: #ffffff;
            font-weight: 600;
            padding-top: 5px;
            padding-bottom: 5px;
            display: inline-block;
            border-radius: 3px;
            margin-right: 15px;
        }
        .btn1:hover{
          background:linear-gradient(135deg,black,darkslategrey);
         box-shadow: 0px 0px 10px 4px black;
          color: #a5c422;
        }
        .class1{
            font-weight: bold;
            font-size: 25px;
            font-family:monospace;
        }

        .output{
            margin-top: auto;
            margin-bottom: auto;
            margin-left: 20px;
            margin-right: 20px;
        }
    </style>
</head>
<body id="top" data-spy="scroll" data-target=".navbar-collapse" data-offset="50">

     <!-- PRE LOADER -->
     <section class="preloader">
          <div class="spinner">

               <span class="spinner-rotate"></span>

          </div>
     </section>


     <!-- HEADER -->
     <header>
          <div class="container">
               <div class="row">

                    <div class="col-md-4 col-sm-5">
                         <p>Welcome to Health Care Bot</p>
                    </div>

                    <div class="col-md-8 col-sm-7 text-align-right">
                         <span class="phone-icon"><i class="fa fa-phone"></i> +91-010-060-0160</span>
                         <span class="email-icon"><i class="fa fa-envelope-o"></i> <a href="#">stl@company.com</a></span>
                    </div>

               </div>
          </div>
     </header>

    <!-- MENU -->
     <section class="navbar navbar-default navbar-static-top" role="navigation">
          <div class="container">

               <div class="navbar-header">
                    <button class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
                         <span class="icon icon-bar"></span>
                         <span class="icon icon-bar"></span>
                         <span class="icon icon-bar"></span>

                    </button>

                    <!-- lOGO TEXT HERE -->
                    <a href="home.php" class="navbar-brand"><i class="fa fa-h-square"></i>ealth Care Bot</a>
               </div>

               <!-- MENU LINKS -->
               <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                         <li><a href="home.php" class="smoothScroll">Home</a></li>
                         <li><a href="#idlast" class="smoothScroll">Contact</a></li>
                        <li><form method="post"><button type="submit"name="logout" class="btn1">LogOut</button></form></li>
                        <li><h5><i class="fas fa-user"></i><?php echo $_SESSION['username'];?></h5></li>
                    </ul>
               </div>

          </div>
     </section>
<!-- DONOR FORM-->
     <div class="main-w3layouts wrapper">
		<h2 align="center">Blood Donation Form</h2>
		<div class="main-agileinfo">
			<div class="agileits-top">
				<form action="#" method="post">
					<input class="text" type="text" name="name" placeholder="Name" required=""><br>
                    <input class="text" type="text" name="gender" placeholder="Gender" required=""><br>
                    <input class="text" type="text" name="age" placeholder="Age" required=""><br>
                    <input class="text" type="text" name="address" placeholder="Address" required=""><br>
					<input class="text" type="text" name="city" placeholder="City" required=""><br>
					<input class="text" type="text" name="disease" placeholder="Disease if any" required=""><br>
					<input class="text" type="text" name="blood_group" placeholder="Blood Group" required=""><br>
                    <input class="text" type="text" name="contact" placeholder="Contact" required=""><br>
					<div class="wthree-text">
						<label class="anim">
							<input type="checkbox" class="checkbox" required="">
							<span>I Agree To The Terms & Conditions</span>
						</label>
						<div class="clear"> </div>
					</div>
					<input type="submit" onclick="alert('Thank You! Your data has been recorded' )" name="submit3" value="SUBMIT">
				</form>
			</div>
		</div>


	</div>
    <!-- FOOTER -->
    <div >
     <footer data-stellar-background-ratio="5">
          <div class="container" id="idlast">
               <div class="row">

                    <div >
                         <div class="footer-thumb">
                             <h4 class="wow fadeInUp" data-wow-delay="0.4s"><p class="contact-info"><h4>Contact Info</h4></p></h4>
                              <p>STL Hospital<br>MANIT Square-462003<br>Bhopal, M.P</p>

                              <div class="contact-info">
                                   <p><i class="fa fa-phone"></i> +91-010-070-0170</p>
                                   <p><i class="fa fa-envelope-o"></i> <a href="#">stl@company.com</a></p>
                              </div>
                         </div>
                    </div>




                    <div class="col-md-12 col-sm-12 border-top">
                         <div class="col-md-4 col-sm-6">
                              <div class="copyright-text">
                                   <p>Copyright &copy; 2020 STL </p>
                              </div>
                         </div>
                         <div class="col-md-6 col-sm-6">
                              <div class="footer-link">
                                   <a href="#">STL Laboratory Tests</a>
                                   <a href="#">Departments</a>
                                   <a href="#">Insurance Policy</a>
                                   <a href="#">Careers</a>
                              </div>
                         </div>
                         <div class="col-md-2 col-sm-2 text-align-center">
                              <div class="angle-up-btn">
                                  <a href="#top" class="smoothScroll wow fadeInUp" data-wow-delay="1.2s"><i class="fa fa-angle-up"></i></a>
                              </div>
                         </div>
                    </div>

               </div>


     </footer>
    </div>
    <!-- SCRIPTS -->
     <script src="js/jquery.js"></script>
     <script src="js/bootstrap.min.js"></script>
     <script src="js/jquery.sticky.js"></script>
     <script src="js/jquery.stellar.min.js"></script>
     <script src="js/wow.min.js"></script>
     <script src="js/smoothscroll.js"></script>
     <script src="js/owl.carousel.min.js"></script>
     <script src="js/custom.js"></script>

</body>
</html>
