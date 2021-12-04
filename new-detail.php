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
     <title>Symptoms</title>
     <style>
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
     </style>
     <meta charset="UTF-8">
     <meta http-equiv="X-UA-Compatible" content="IE=Edge">
     <meta name="description" content="">
     <meta name="keywords" content="">
     <meta name="author" content="Tooplate">
     <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">

     <link rel="stylesheet" href="css/bootstrap.min.css">
     <link rel="stylesheet" href="css/magnific-popup.css">

     <link rel="stylesheet" href="css/font-awesome.min.css">
     <link rel="stylesheet" href="css/animate.css">

     <link rel="stylesheet" href="css/owl.carousel.css">
     <link rel="stylesheet" href="css/owl.theme.default.min.css">

     <!-- MAIN CSS -->
     <link rel="stylesheet" href="css/tooplate-style.css">
     <style type="text/css">
        #id1{
            width:0;
            height:0;
            position:fixed;
            overflow:hidden;
            top:0;
            left:0;
            transition:all 1s ease-in-out;
        }
       #id1:target{
            height:90%;
            width:90%;
            top:10%;
            left:5%;
            background-color:cyan;
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
                    <a href="home.php" class="navbar-brand"><i class="fa fa-c-square"></i>ChooseLyf</a>
               </div>

               <!-- MENU LINKS -->
               <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                         <li><a href="home.php" class="smoothScroll">Home</a></li>
                         <li><a href="#idb" class="smoothScroll">Contact</a></li>
                        <li><form method="post"><button type="submit"name="logout" class="btn1">LogOut</button></form></li>
                        <li><h5><i class="fas fa-user"></i><?php echo $_SESSION[
                            'username'
                        ]; ?></h5></li>
                    </ul>
               </div>

          </div>
     </section>


     <!-- SYMPTOMS detail -->
     <section id="news-detail" data-stellar-background-ratio="0.5">
          <div class="container">
               <div class="row">

                    <div class="col-md-8 col-sm-7">
                         <!-- NEWS THUMB -->
                         <div class="news-detail-thumb">
                              <div class="news-image">
                                   <img src="https://img.graphicsurf.com/2020/08/medicine-vector-flat-illustration.jpg" class="img-responsive" alt="">
                              </div>
                              <h3>Select the symptoms you are experiencing...</h3>
                              <blockquote>It is the important part for judgement.So select wisely.</blockquote>
                              <form  action="index.php" method="post">
                                <input type="checkbox" id="symptom1" name="symptom1" value="abdominal_pain">
                                  <label for="symptom1"> Abdominal_pain</label><br>
                                  <input type="checkbox" id="symptom2" name="symptom2" value="abnormal_menstruation">
                                  <label for="symptom2"> Abnormal_menstruation</label><br>
                                  <input type="checkbox" id="symptom3" name="symptom3" value="acidity	">
                                  <label for="symptom3"> Acidity	</label><br>
                                  <input type="checkbox" id="symptom4" name="symptom4" value="acute_liver_failure">
                                  <label for="symptom4"> Acute_liver_failure</label><br>
                                  <input type="checkbox" id="symptom5" name="symptom5" value="altered_sensorium">
                                  <label for="symptom5"> Altered_sensorium</label><br>
                                  <input type="checkbox" id="symptom6" name="symptom6" value="anxiety">
                                  <label for="symptom6"> Anxiety	</label><br>
                                  <input type="checkbox" id="symptom7" name="symptom7" value="back_pain">
                                  <label for="symptom7"> Back_pain</label><br>
                                  <input type="checkbox" id="symptom8" name="symptom8" value="belly_pain">
                                  <label for="symptom8"> Belly_pain</label><br>
                                  <input type="checkbox" id="symptom9" name="symptom9" value="blackheads	">
                                  <label for="symptom9"> Blackheads	</label><br>
                                  <input type="checkbox" id="symptom10" name="symptom10" value="bladder_discomfort">
                                  <label for="symptom10"> Bladder_discomfort</label><br>
                                  <input type="checkbox" id="symptom11" name="symptom11" value="blister">
                                  <label for="symptom11"> Blister</label><br>
                                  <input type="checkbox" id="symptom12" name="symptom12" value="blood_in_sputum	">
                                  <label for="symptom12"> Blood_in_sputum	</label><br>
                                  <input type="checkbox" id="symptom13" name="symptom13" value="bloody_stool">
                                  <label for="symptom13"> Bloody_stool</label><br>
                                  <input type="checkbox" id="symptom14" name="symptom14" value="blood_and_distorted_vision">
                                  <label for="symptom14"> Blood_and_distorted_vision</label><br>
                                  <input type="checkbox" id="symptom15" name="symptom15" value="breathlessness">
                                  <label for="symptom15"> Breathlessness	</label><br>
                                <input type="checkbox" id="symptom16" name="symptom16" value="brittle_nails">
                                  <label for="symptom16"> Brittle_nails</label><br>
                                  <input type="checkbox" id="symptom17" name="symptom17" value="bruising">
                                  <label for="symptom17"> Bruising</label><br>
                                  <input type="checkbox" id="symptom18" name="symptom18" value="burning_micturition	">
                                  <label for="symptom18"> Burning_micturition	</label><br>
                                <input type="checkbox" id="symptom19" name="symptom19" value="chest_pain">
                                  <label for="symptom19"> Chest_pain</label><br>
                                  <input type="checkbox" id="symptom20" name="symptom20" value="chills	">
                                  <label for="symptom20"> Chills	</label><br>
                                  <input type="checkbox" id="symptom21" name="symptom21" value="cold_hands_and_feets		">
                                  <label for="symptom21"> Cold_hands_and_feets		</label><br>
                                <input type="checkbox" id="symptom22" name="symptom22" value="coma	">
                                  <label for="symptom22"> Coma	</label><br>
                                  <input type="checkbox" id="symptom23" name="symptom23" value="congestion">
                                  <label for="symptom23">Congestion</label><br>
                                  <input type="checkbox" id="symptom24" name="symptom24" value="constipation	">
                                  <label for="symptom24"> Constipation	</label><br>
                                <input type="checkbox" id="symptom25" name="symptom25" value="continuous_feel_of_urine">
                                  <label for="symptom25"> Continuous_feel_of_urine</label><br>
                                  <input type="checkbox" id="symptom26" name="symptom26" value="continuous_sneezing">
                                  <label for="symptom26"> Continuous_sneezing</label><br>
                                  <input type="checkbox" id="symptom27" name="symptom27" value="cough		">
                                  <label for="symptom27"> Cough		</label><br>
                                <input type="checkbox" id="symptom28" name="symptom28" value="cramps	">
                                  <label for="symptom28"> Cramps	</label><br>
                                  <input type="checkbox" id="symptom29" name="symptom29" value="dark_urine">
                                  <label for="symptom29"> Dark_urine</label><br>
                                  <input type="checkbox" id="symptom30" name="symptom30" value="dehydration	">
                                  <label for="symptom30">Dehydration	</label><br>
                                <input type="checkbox" id="symptom31" name="symptom31" value="depression">
                                  <label for="symptom31"> Depression</label><br>
                                  <input type="checkbox" id="symptom32" name="symptom32" value="diarrhoea">
                                  <label for="symptom32"> Diarrhoea</label><br>
                                  <input type="checkbox" id="symptom33" name="symptom33" value="dischromic _patches	">
                                  <label for="symptom33"> Dischromic _patches	</label><br>
                                <input type="checkbox" id="symptom34" name="symptom34" value="distention_of_abdomen">
                                  <label for="symptom34"> Distention_of_abdomen</label><br>
                                  <input type="checkbox" id="symptom35" name="symptom35" value="dizziness">
                                  <label for="symptom35"> Dizziness</label><br>
                                  <input type="checkbox" id="symptom36" name="symptom36" value="drying_and_tingling_lips		">
                                  <label for="symptom36">Drying_and_tingling_lips		</label><br>
                                <input type="checkbox" id="symptom37" name="symptom37" value="enlarged_thyroid">
                                  <label for="symptom37"> Enlarged_thyroid</label><br>
                                  <input type="checkbox" id="symptom38" name="symptom38" value="excessive_hunger">
                                  <label for="symptom38"> Excessive_hunger</label><br>
                                  <input type="checkbox" id="symptom39" name="symptom39" value="extra_marital_contacts		">
                                  <label for="symptom39"> Extra_marital_contacts		</label><br>
                                <input type="checkbox" id="symptom40" name="symptom40" value="family_history">
                                  <label for="symptom40">Family_history</label><br>
                                  <input type="checkbox" id="symptom41" name="symptom41" value="fast_heart_rate	">
                                  <label for="symptom41"> Fast_heart_rate	</label><br>
                                  <input type="checkbox" id="symptom42" name="symptom42" value="fatigue		">
                                  <label for="symptom42"> Fatigue		</label><br>
                                  <input type="checkbox" id="symptom43" name="symptom43" value="fluid_overload	">
                                  <label for="symptom43"> Fluid_overload	</label><br>

                                  <!-- <input type="checkbox" id="symptom44" name="symptom44" value="fluid_overload	">
                                  <label for="symptom44"> Fluid_overload_Excess	</label><br> -->
                                  <input type="checkbox" id="symptom45" name="symptom45" value="foul_smell_of_urine">
                                  <label for="symptom45"> Foul_smell_of_urine</label><br>
                                  <input type="checkbox" id="symptom46" name="symptom46" value="headache		">
                                  <label for="symptom46"> Headache		</label><br>
                                   <input type="checkbox" id="symptom47" name="symptom47" value="high_fever	">
                                  <label for="symptom47"> High_fever	</label><br>
                                  <input type="checkbox" id="symptom48" name="symptom48" value="hip_joint_pain	">
                                  <label for="symptom48"> Hip_joint_pain	</label><br>
                                  <input type="checkbox" id="symptom49" name="symptom49" value="history_of_alcohol_consumption	">
                                  <label for="symptom49"> History_of_alcohol_consumption		</label><br>
                                  <input type="checkbox" id="symptom50" name="symptom50" value="increased_appetite">
                                  <label for="symptom50"> Increased_appetite</label><br>
                                  <input type="checkbox" id="symptom51" name="symptom51" value="indigestion">
                                  <label for="symptom51"> Indigestion</label><br>
                                  <input type="checkbox" id="symptom52" name="symptom52" value="inflammatory_nails">
                                  <label for="symptom52"> Inflammatory_nails</label><br>
                                  <input type="checkbox" id="symptom53" name="symptom53" value="internal_itching">
                                  <label for="symptom53">Internal_itching</label><br>
                                  <input type="checkbox" id="symptom54" name="symptom54" value="irregular_sugar_level">
                                  <label for="symptom54"> Irregular_sugar_level</label><br>
                                  <input type="checkbox" id="symptom55" name="symptom55" value="irritability">
                                  <label for="symptom55"> Irritability	</label><br>
                                  <input type="checkbox" id="symptom56" name="symptom56" value="irritation_in_anus">
                                  <label for="symptom56"> Irritation_in_anus</label><br>
                                  <input type="checkbox" id="symptom57" name="symptom57" value="itching	">
                                  <label for="symptom57"> Itching	</label><br>
                                  <input type="checkbox" id="symptom58" name="symptom58" value="joint_pain">
                                  <label for="symptom58"> Joint_pain	</label><br>
                                  <input type="checkbox" id="symptom59" name="symptom59" value="knee_pain">
                                  <label for="symptom59"> knee_pain</label><br>
                                  <input type="checkbox" id="symptom60" name="symptom60" value="lack_of_concentration">
                                  <label for="symptom60"> Lack_of_concentration</label><br>
                                  <input type="checkbox" id="symptom61" name="symptom61" value="lethargy">
                                  <label for="symptom61"> Lethargy	</label><br>
                                <input type="checkbox" id="symptom62" name="symptom62" value="loss_of_appetite">
                                  <label for="symptom62"> Loss_of_appetite</label><br>
                                  <input type="checkbox" id="symptom63" name="symptom63" value="loss_of_balance	">
                                  <label for="symptom63"> Loss_of_balance	</label><br>
                                  <input type="checkbox" id="symptom64" name="symptom64" value="loss_of_smell	">
                                  <label for="symptom64">Loss_of_smell	</label><br>
                                <input type="checkbox" id="symptom65" name="symptom65" value="malaise">
                                  <label for="symptom65">Malaise</label><br>
                                  <input type="checkbox" id="symptom66" name="symptom66" value="mild_fever">
                                    <label for="symptom66">Mild_fever</label><br>
                                  <input type="checkbox" id="symptom67" name="symptom67" value="mood_swings">
                                  <label for="symptom67">Mood_swings	</label><br>
                                  <input type="checkbox" id="symptom68" name="symptom68" value="movement_stiffness">
                                  <label for="symptom68"> Movement_stiffness</label><br>
                                <input type="checkbox" id="symptom69" name="symptom69" value="mucoid_sputum	">
                                  <label for="symptom69"> Mucoid_sputum	</label><br>
                                  <input type="checkbox" id="symptom70" name="symptom70" value="muscle_pain	">
                                  <label for="symptom70"> Muscle_pain	</label><br>
                                  <input type="checkbox" id="symptom71" name="symptom71" value="muscle_wasting		">
                                  <label for="symptom71"> Muscle_wasting		</label><br>
                                <input type="checkbox" id="symptom72" name="symptom72" value="muscle_weakness	">
                                  <label for="symptom72"> Muscle_weakness	</label><br>
                                  <input type="checkbox" id="symptom73" name="symptom73" value="nausea	">
                                  <label for="symptom73">Nausea	</label><br>
                                  <input type="checkbox" id="symptom74" name="symptom74" value="neck_pain	">
                                  <label for="symptom74">Neck_pain	</label><br>
                                <input type="checkbox" id="symptom75" name="symptom75" value="nodal_skin_eruptions">
                                  <label for="symptom75"> Nodal_skin_eruptions</label><br>
                                  <input type="checkbox" id="symptom76" name="symptom76" value="obesity">
                                  <label for="symptom76"> Obesity	</label><br>
                                  <input type="checkbox" id="symptom77" name="symptom77" value="pain_behind_the_eyes	">
                                  <label for="symptom77"> Pain_behind_the_eyes	</label><br>
                                <input type="checkbox" id="symptom78" name="symptom78" value="pain_during_bowel_movements">
                                  <label for="symptom78"> Pain_during_bowel_movements</label><br>
                                  <input type="checkbox" id="symptom79" name="symptom79" value="pain_in_anal_region">
                                  <label for="symptom79">Pain_in_anal_region</label><br>
                                  <input type="checkbox" id="symptom80" name="symptom80" value="painful_walking">
                                  <label for="symptom80">Painful_walking</label><br>
                                  <input type="checkbox" id="symptom81" name="symptom81" value="palpitations">
                                  <label for="symptom81">palpitations</label><br>
                                <input type="checkbox" id="symptom82" name="symptom82" value="passage_of_gases">
                                  <label for="symptom82">Passage_of_gases</label><br>
                                  <input type="checkbox" id="symptom83" name="symptom83" value="patches_in_throat">
                                  <label for="symptom83">  Patches_in_throat</label><br>
                                  <input type="checkbox" id="symptom84" name="symptom84" value="phlegm">
                                  <label for="symptom84">Phlegm</label><br>
                                  <input type="checkbox" id="symptom85" name="symptom85" value="polyuria">
                                  <label for="symptom85">polyuria</label><br>
                                <input type="checkbox" id="symptom86" name="symptom86" value="prominent_veins_on_calf">
                                  <label for="symptom86"> Prominent_veins_on_calf</label><br>
                                  <input type="checkbox" id="symptom87" name="symptom87" value="puffy_face_and_eyes">
                                  <label for="symptom87">Puffy_face_and_eyes</label><br>
                                  <input type="checkbox" id="symptom88" name="symptom88" value="pus_filled_pimples">
                                  <label for="symptom88"> Pus_filled_pimples	</label><br>
                                <input type="checkbox" id="symptom89" name="symptom89" value="receiving_blood_transfusion">
                                  <label for="symptom89">Receiving_blood_transfusion</label><br>
                                  <input type="checkbox" id="symptom90" name="symptom90" value="receiving_unsterile_injections">
                                  <label for="symptom90"> Receiving_unsterile_injections</label><br>
                                  <input type="checkbox" id="symptom91" name="symptom91" value="red_sore_around_nose	">
                                  <label for="symptom91"> Red_sore_around_nose		</label><br>
                                <input type="checkbox" id="symptom92" name="symptom92" value="red_spots_over_body">
                                  <label for="symptom92">Red_spots_over_body</label><br>
                                  <input type="checkbox" id="symptom93" name="symptom93" value="redness_of_eyes">
                                  <label for="symptom93"> Redness_of_eyes</label><br>
                                  <input type="checkbox" id="symptom94" name="symptom94" value="	restlessness">
                                  <label for="symptom94">restlessness</label><br>
                                  <input type="checkbox" id="symptom95" name="symptom95" value="runny_nose">
                                  <label for="symptom95">Runny_nose</label><br>
                                  <input type="checkbox" id="symptom96" name="symptom96" value="rusty_sputum">
                                  <label for="symptom96">Rusty_sputum</label><br>
                                  <input type="checkbox" id="symptom97" name="symptom97" value="scurring">
                                  <label for="symptom97">Scurring</label><br>
                                  <input type="checkbox" id="symptom98" name="symptom98" value="shivering">
                                  <label for="symptom98">Shivering</label><br>
                                  <input type="checkbox" id="symptom99" name="symptom99" value="silver_like_dusting">
                                  <label for="symptom99">Silver_like_dusting</label><br>
                                  <input type="checkbox" id="symptom100" name="symptom100" value="sinus_pressure">
                                  <label for="symptom100">Sinus_pressure</label><br>
                                  <input type="checkbox" id="symptom101" name="symptom101" value="skin_peeling">
                                  <label for="symptom101">Skin_peeling</label><br>
                                  <input type="checkbox" id="symptom102" name="symptom102" value="skin_rash">
                                  <label for="symptom102">Skin_rash</label><br>
                                  <input type="checkbox" id="symptom103" name="symptom103" value="slurred_speech">
                                  <label for="symptom103">Slurred_speech	</label><br>
                                  <input type="checkbox" id="symptom104" name="symptom104" value="small_dents_in_nails">

                                  <label for="symptom104">Small_dents_in_nails</label><br>

                                  <input type="checkbox" id="symptom105" name="symptom105" value="spinning_movements">
                                  <label for="symptom105">Spinning_movements</label><br>
                                  <input type="checkbox" id="symptom106" name="symptom106" value="spotting_ urination">
                                  <label for="symptom106">Spotting_ urination</label><br>
                                  <input type="checkbox" id="symptom107" name="symptom107" value="stiff_neck">
                                  <label for="symptom107">Stiff_neck</label><br>
                                  <input type="checkbox" id="symptom108" name="symptom108" value="stomach_bleeding">
                                  <label for="symptom108">Stomach_bleeding</label><br>
                                  <input type="checkbox" id="symptom109" name="symptom109" value="stomach_pain">
                                  <label for="symptom109">Stomach_pain</label><br>
                                  <input type="checkbox" id="symptom110" name="symptom110" value="sunken_eyes">
                                  <label for="symptom110">Sunken_eyes</label><br>
                                <input type="checkbox" id="symptom111" name="symptom111" value="sweating">
                                  <label for="symptom111">Sweating</label><br>
                                  <input type="checkbox" id="symptom112" name="symptom112" value="swelled_lymph_nodes">
                                  <label for="symptom112">Swelled_lymph_nodes</label><br>
                                  <input type="checkbox" id="symptom113" name="symptom113" value="swelling_joints">
                                  <label for="symptom113">Swelling_joints	</label><br>
                                <input type="checkbox" id="symptom114" name="symptom114" value="swelling_of_stomach">
                                  <label for="symptom114">Swelling_of_stomach</label><br>
                                  <input type="checkbox" id="symptom115" name="symptom115" value="swollen_blood_vessels">
                                  <label for="symptom115">Swollen_blood_vessels</label><br>
                                  <input type="checkbox" id="symptom116" name="symptom116" value="swollen_extremeties">
                                  <label for="symptom116">Swollen_extremeties</label><br>
                                <input type="checkbox" id="symptom117" name="symptom117" value="swollen_legs">
                                  <label for="symptom117">Swollen_legs</label><br>
                                  <input type="checkbox" id="symptom118" name="symptom118" value="throat_irritation">
                                  <label for="symptom118">Throat_irritation</label><br>
                                  <input type="checkbox" id="symptom119" name="symptom119" value="toxic_look_(typhos)">
                                  <label for="symptom119">Toxic_look_(typhos)</label><br>
                                <input type="checkbox" id="symptom120" name="symptom120" value="ulcers_on_tongue">
                                  <label for="symptom120">Ulcers_on_tongue</label><br>
                                  <input type="checkbox" id="symptom121" name="symptom121" value="unsteadiness">
                                  <label for="symptom121">Unsteadiness</label><br>
                                  <input type="checkbox" id="symptom122" name="symptom122" value="visual_disturbances">
                                  <label for="symptom122">Visual_disturbances</label><br>
                                <input type="checkbox" id="symptom123" name="symptom123" value="vomiting">
                                  <label for="symptom123">Vomiting</label><br>
                                  <input type="checkbox" id="symptom124" name="symptom124" value="watering_from_eyes">
                                  <label for="symptom124">Watering_from_eyes</label><br>
                                  <input type="checkbox" id="symptom125" name="symptom125" value="weakness_in_limbs">
                                  <label for="symptom125">Weakness_in_limbs</label><br>
                                <input type="checkbox" id="symptom126" name="symptom126" value="weakness_of_one_body_side">
                                  <label for="symptom126">Weakness_of_one_body_side</label><br>
                                  <input type="checkbox" id="symptom127" name="symptom127" value="weight_gain">
                                  <label for="symptom127">Weight_gain</label><br>
                                  <input type="checkbox" id="symptom128" name="symptom128" value="weight_loss">
                                  <label for="symptom128">Weight_loss</label><br>
                                <input type="checkbox" id="symptom129" name="symptom129" value="yellow_crust_ooze">
                                  <label for="symptom129">Yellow_crust_ooze</label><br>
                                  <input type="checkbox" id="symptom130" name="symptom130" value="yellow_urine">
                                  <label for="symptom130">Yellow_urine</label><br>
                                  <input type="checkbox" id="symptom131" name="symptom131" value="yellowing_of_eyes">
                                  <label for="symptom131">Yellowing_of_eyes</label><br>
                                <input type="checkbox" id="symptom132" name="symptom132" value="yellowish_skin">
                                  <label for="symptom132"> Yellowish_skin</label><br>
                                  <input  type="submit" name="submit" class="btn1" value="Submit">

                                            </form>
                         </div>
                    </div>
                    <section>
                        <div class="container">
                                <div>
                                  <h2 class="wow fadeInUp" data-wow-delay="0.6s">Modern healthcare today
</h2>
                                </div>
                                <div class="wow fadeInUp" data-wow-delay="0.8s">
                                         <p>Share to Twitter
Modern healthcare today
Modern healthcare has been transformed by advances in technology. The rise of a consumerization culture where people can easily shop around for suitable services also has an impact on modern healthcare. .<br></p>
                                </div>


                        </div>
                    </section>


               </div>
          </div>
     </section>



     <!-- FOOTER -->
     <footer data-stellar-background-ratio="5">
          <div class="container" id="idb">
               <div class="row">

                    <div class="col-md-4 col-sm-6">
                         <div class="footer-thumb">
                              <h4 class="wow fadeInUp" data-wow-delay="0.4s">Contact Info</h4>
                              <p>ChooseLyf Limited<br>College Square-745896<br>Kolkata, W.B.</p>

                              <div class="contact-info">
                                   <p><i class="fa fa-phone"></i> +91-9330554877</p>
                                   <p><i class="fa fa-envelope-o"></i> <a href="#">chooselyf@gmail.com</a></p>
                              </div>
                         </div>
                    </div>




                    <div class="col-md-12 col-sm-12 border-top">
                         <div class="col-md-4 col-sm-6">
                              <div class="copyright-text">
                                   <p>Copyright &copy; 2021 ChooseLyf</p>
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
     <script src="js/jquery.magnific-popup.min.js"></script>
     <script src="js/magnific-popup-options.js"></script>
     <script src="js/wow.min.js"></script>
     <script src="js/smoothscroll.js"></script>
     <script src="js/owl.carousel.min.js"></script>
     <script src="js/custom.js"></script>

</body>
</html>
