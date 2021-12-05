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
     <title>Home</title>
     <style>
     .textalign{
         padding-top:30px;
         padding-bottom: 30px;
         background-color: darkseagreen;
         text-align: center;
         color: white;

     }
     .textalign:hover{
         background-color: antiquewhite;
     }
     .textalign1:hover{
         color:lightcoral;
         font-size:52px;
     }
     .textalign1{
         text-align: center;
         color: white;
         font-size:50px;
     }
     .textalign2:hover{
         color:lightcoral;
         font-size: 17px;
     }
     .textalign2{
         text-align: center;
         color: white;
         font-size:15px;
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
     .wow {
          color: #a5c422;
     }
     </style>
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

                         <span class="phone-icon"><i class="fa fa-phone"></i> +</span>
                         <span class="email-icon"><i class="fa fa-envelope-o"></i> <a href="#">stl@company.com</a></span>
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
                    <a href="home.php" class="navbar-brand"><i class="fa fa-h-square"></i>ChooseLyf</a>
               </div>

               <!-- MENU LINKS -->
               <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                         <li><a href="#top" class="smoothScroll">Home</a></li>
                         <li><a href="#id1"  class="smoothScroll">Contact</a></li>
                        <li><form method="post"><button type="submit" name="logout" class="btn1">LogOut</button></form></li>
                      <li><h5><i class="fas fa-user"></i><?php echo $_SESSION[
                          'username'
                      ]; ?></h5></li>
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
<!--                                             <h3>Aenean luctus lobortis tellus</h3>-->
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
                              <h2 class="wow fadeInUp" data-wow-delay="0.6s">Welcome to  <i class="fa fa-c-square"></i>ChooseLyf App</h2>
                              <div class="wow fadeInUp" data-wow-delay="0.8s">
                                   <!-- <p>Improve health and care with us</p> -->

                              </div>
                              <figure class="profile wow fadeInUp" data-wow-delay="1s">
                                   <img src="https://image.freepik.com/free-vector/medical-background-design_1212-116.jpg" class="img-responsive" alt="">
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

     <section>
         <div class="textalign">
                 <div>
                     <h2 class="wow fadeInUp" data-wow-delay="0.2s"><p class="textalign1">Modern healthcare today</p></h2>
                 </div>
                 <div class="wow fadeInUp" data-wow-delay="0.4s">
                          <p class="textalign2">Share to Twitter
Modern healthcare today
Modern healthcare has been transformed by advances in technology. The rise of a consumerization culture where people can easily shop around for suitable services also has an impact on modern healthcare.<br></p>
                 </div>


         </div>
    </section>

     <!-- HOW IT WORKS -->

     <section id="team" data-stellar-background-ratio="1">
          <div class="container">
               <div class="row">

                    <div class="col-md-6 col-sm-6">
                         <div class="about-info">
                              <h2 class="wow fadeInUp" data-wow-delay="0.1s">How it works</h2>
                         </div>
                    </div>

                    <div class="clearfix"></div>

                    <div class="col-md-4 col-sm-6">
                         <div class="team-thumb wow fadeInUp" data-wow-delay="0.2s">
                             <a href="index.php"><img src="images/Check.png" class="img-responsive" alt=""></a>
                             <div  class="team-contact-info" >
                                 <p>Get regular checkups.</p>
                             </div>



                         </div>
                    </div>

                    <div class="col-md-4 col-sm-6">
                         <div class="team-thumb wow fadeInUp" data-wow-delay="0.4s">
                             <a href="index.php"><img src="images/Clock.png" class="img-responsive" alt=""></a>
                              <div class="team-contact-info" >
                                  <p>Book appointments.</p>
                             </div>
                         </div>
                    </div>

                    <div class="col-md-4 col-sm-6">
                         <div class="team-thumb wow fadeInUp" data-wow-delay="0.6s">
                             <a href="blood.php"><img src="images/Heart.png" class="img-responsive" alt=""></a>
                                <div class="team-contact-info">
                                    <p>Check the blood availability .</p>
                             </div>
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
                              <h4 class="wow fadeInUp" data-wow-delay="0.4s">Contact Info</h4>
                              <p>ChooseLyf Limited<br>COLLEGE Square-754214<br>Kolkata, W.B.</p>

                              <div class="contact-info">
                                   <p><i class="fa fa-phone"></i> +91-9330554877</p>
                                   <p><i class="fa fa-envelope-o"></i> <a href="#">chooselyf@gmail.com</a></p>
                              </div>
                         </div>
                    </div>




                    <div class="col-md-12 col-sm-12 border-top">
                         <div class="col-md-4 col-sm-6">
                              <div class="copyright-text">
                                   <p>Copyright &copy; 2021 ChooseLyf </p>
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
