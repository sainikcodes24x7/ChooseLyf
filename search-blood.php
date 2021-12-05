<?php
session_start();
if(isset($_POST['logout']))
{include("destroy.php");}
if($_SESSION['status']!="Active")
 {header("location:login.php");}
if(isset($_POST['searchBtn']))
{
  $bg=$_POST['blood_group'];
	$servername='localhost';
	$user='root';
	 $password='';
	$db='register';
	$con=mysqli_connect($servername,$user,$password,$db);
	if(!$con)
	{
		echo "Unable to connect to database".mysqli_connect_error();
	}
  $query="SELECT * FROM donar WHERE Bloodgroup='$bg'";
	$result=mysqli_query($con,$query);
}
?>
<!DOCTYPE html>
<html>
<head>
  <script src="https://kit.fontawesome.com/b62e73b8ee.js" crossorigin="anonymous"></script>
<title>Find Donor</title>

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

     <!-- MAIN CSS -->
     <link rel="stylesheet" href="css/tooplate-style.css">
    <style>
   table,th,td{
     border: 2px solid black;
     padding: 10px;
   }
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
                         <li><a href="blood.php" class="smoothScroll">Contact</a></li>
                        <li><form method="post"><button type="submit"name="logout" class="btn1">LogOut</button></form></li>
                        <li><h5><i class="fas fa-user"></i><?php echo $_SESSION['username'];?></h5></li>
                    </ul>
               </div>

          </div>
     </section>
<!--    SEARCH BUTTON-->
     <section>
         <div class="searchopt">
        <div class="col-md-6">
            <div class="form-group col-md-12">
                <form class="form-horizontal" method="post" action="search-blood.php">
                    <div class="form-group">
                        <label class="col-sm-6" ><p class="class1">Search for donor with blood group</p></label>
                        <div class="col-sm-4">
                            <select name="blood_group" class="form-control" >
                              <option value="A+">A+</option>
                                <option value="O+">O+</option>
                                <option value="O-">O-</option>
                                <option value="A-">A-</option>
                                <option value="B+">B+</option>
                                <option value="B-">B-</option>
                                <option value="AB+">AB+</option>
                                <option value="AB-">AB-</option>
                            </select>
                        </div>
                        <div class="col-sm-2">
                            <button class="btn1" name="searchBtn" >Search</button>
                        </div>
                    </div>

                </form>
            </div>
        </div>
    </div>
    </section>
    <!-- printing the table-->
    <section>
        <div class="output" >
    <?php if(isset($_POST['searchBtn'])){?>
    <table border="2" width="1500">
     <tr>
       <td>Serial No.</td>
       <td>Name</td>
       <td>Gender</td>
       <td>Age</td>
       <td>Address</td>
       <td>City</td>
       <td>Disease if any</td>
       <td>BloodGroup</td>
       <td>Contact</td>
     </tr>
     <?php
     $x=1;
     while($feed=mysqli_fetch_assoc($result)):?>
     <tr>
       <td><?php echo $x?></td>
       <td><?php echo $feed['Name']?></td>
       <td><?php echo $feed['Gender']?></td>
       <td><?php echo $feed['Age']?></td>
       <td><?php echo $feed['Address']?></td>
       <td><?php echo $feed['City']?></td>
       <td><?php echo $feed['Disease']?></td>
       <td><?php echo $feed['Bloodgroup']?></td>
       <td><?php echo $feed['Contact']?></td>
       <?php $x=$x+1;?>
     </tr>
   <?php endwhile; }?>
    <!--    SPACE FOR OUTPUT-->

        </div>
    </section>

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
