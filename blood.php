<?php
session_start();
if (isset($_POST['logout'])) {
     include 'destroy.php';
}
if ($_SESSION['status'] != 'Active') {
     header('location:login.php');
}
?>
<!DOCTYPE html>
<html lang="en">

<head>
     <script src="https://kit.fontawesome.com/b62e73b8ee.js" crossorigin="anonymous"></script>
     <title>Blood Bank</title>

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
          .btn1 {
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

          .btn1:hover {
               background: linear-gradient(135deg, black, darkslategrey);
               box-shadow: 0px 0px 10px 4px black;
               color: #a5c422;
          }

          .class1 {
               position: relative;
               width: 70%;
               left: 5%;
               background-color: #a5c422;
               text-align: center;
               border-radius: 3px;
               border: 2px solid darkslategrey;
               margin: 5px;
               margin-bottom: 3vw;
               padding: 10px;
               transition: background 0.2s, box-shadow 0.5s;
               font-weight: bold;
               color: white;
               margin-left: 10%;
               margin-top: 40%;
          }

          .class1:hover {
               background: linear-gradient(135deg, black, darkslategrey);
               color: #a5c422;
               box-shadow: 0px 0px 10px 4px black;

          }

          .wow {
               color: #a5c422;
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
                         <p>Welcome to ChooseLyf</p>
                    </div>

                    <div class="col-md-8 col-sm-7 text-align-right">
                         <span class="phone-icon"><i class="fa fa-phone"></i> +91-9330554877</span>
                         <span class="email-icon"><i class="fa fa-envelope-o"></i> <a href="#">chooselyf@gmail.com</a></span>
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
                    <a href="home.php" class="navbar-brand"><i class="fa fa-c-square"></i>ChooseLyf App</a>
               </div>

               <!-- MENU LINKS -->
               <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                         <li><a href="home.php" class="smoothScroll">Home</a></li>
                         <li><a href="#id1" class="smoothScroll">Contact</a></li>
                         <li>
                              <form method="post"><button type="submit" name="logout" class="btn1">LogOut</button></form>
                         </li>
                         <li>
                              <h5><i class="fas fa-user"></i><?php echo $_SESSION['username']; ?></h5>
                         </li>
                    </ul>
               </div>

          </div>
     </section>



     <!-- HOME -->
     <section id="home" class="slider" data-stellar-background-ratio="0.5">
          <div class="container">
               <div class="row">

                    <div class="owl-carousel owl-theme">
                         <div class="item item-first">
                              <div class="caption">
                                   <div class="col-md-offset-1 col-md-10">
                                        <h3>Let's make your life happier</h3>
                                        <h1>Healthy Living</h1>

                                   </div>
                              </div>
                         </div>

                         <div class="item item-second">
                              <div class="caption">
                                   <div class="col-md-offset-1 col-md-10">
                                        <h3>There is no one who loves pain itself, who seeks after it and wants to have it, simply because it is pain...</h3>
                                        <h1>New Lifestyle</h1>

                                   </div>
                              </div>
                         </div>

                         <div class="item item-third">
                              <div class="caption">
                                   <div class="col-md-offset-1 col-md-10">
                                        <h3>Pellentesque nec libero nisi</h3>
                                        <h1>Your Health Benefits</h1>
                                   </div>
                              </div>
                         </div>
                    </div>

               </div>
          </div>
     </section>


     <!-- ABOUT -->
     <section id="about">
          <div class="container">
               <div class="row">

                    <div class="col-md-6 col-sm-6">
                         <div class="about-info">
                              <h2 class="wow fadeInUp" data-wow-delay="0.6s">Welcome to <i class="fa fa-c-square"></i>ChooseLyf App</h2>
                              <div class="wow fadeInUp" data-wow-delay="0.8s">
                                   <!-- <p>Improve health and care with us</p> -->

                              </div>
                              <figure class="profile wow fadeInUp" data-wow-delay="1s">
                                   <img src="images/Bot.jpeg" class="img-responsive" alt="">
                                   <figcaption>
                                        <!-- <h3>Dr.Bot</h3>
                                        <p>General Principal</p> -->
                                   </figcaption>
                              </figure>
                         </div>
                    </div>

               </div>
          </div>
     </section>





     <!-- Blood Bank-->
     <section id="predictor" data-stellar-background-ratio="3">
          <div class="container">
               <div class="row">

                    <div class="col-md-6 col-sm-6">
                         <img src="https://us.123rf.com/450wm/yupiramos/yupiramos1803/yupiramos180330947/98841003-heart-and-bag-blood-donor-transfusion-for-medical-care-illustration-.jpg?ver=6" class="img-responsive" alt="">
                    </div>

                    <div class="col-md-6 col-sm-6">
                         <!-- CONTACT FORM HERE -->
                         <div class="col-md-6">

                              <button type="submit" name="search" action="search-blood.php" class="class1">
                                   <a href="search-blood.php">CLICK ME!<br>To Find Donor</a></button>
                              <button type="submit" name="search" class="class1">
                                   <a href="donate.php">CLICK ME!<br>To Donate Blood</a></button>
                         </div>
                    </div>
               </div>
          </div>
     </section>


     <!-- FOOTER -->
     <footer data-stellar-background-ratio="5">
          <div class="container" id="id1">
               <div class="row">

                    <div class="col-md-4 col-sm-6">
                         <div class="footer-thumb">
                              <h4 class="wow fadeInUp" data-wow-delay="0.4s">Contact</h4>
                              <p>STL Hospital<br>COLLEGE Square-754214<br>Kolkata, W.B.</p>

                              <div class="contact-info">
                                   <p><i class="fa fa-phone"></i> +91-9652365874</p>
                                   <p><i class="fa fa-envelope-o"></i> <a href="#">chooselyf@gmail.com</a></p>
                              </div>
                         </div>
                    </div>




                    <div class="col-md-12 col-sm-12 border-top">
                         <div class="col-md-4 col-sm-6">
                              <div class="copyright-text">
                                   <p>Copyright &chooseLyf; 2021 </p>
                              </div>
                         </div>
                         <div class="col-md-6 col-sm-6">
                              <div class="footer-link">
                                   <a href="#">ChooseLyf Tests</a>
                                   <a href="#">Departments</a>
                                   <a href="#">Insurance Policy</a>
                                   <!-- <a href="#">Careers</a> -->
                              </div>
                         </div>
                         <div class="col-md-2 col-sm-2 text-align-center">
                              <div class="angle-up-btn">
                                   <a href="#top" class="smoothScroll wow fadeInUp" data-wow-delay="1.2s"><i class="fa fa-angle-up"></i></a>
                              </div>
                         </div>
                    </div>

               </div>
          </div>
     </footer>

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