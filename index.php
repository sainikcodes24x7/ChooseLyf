<?php
session_start();
if (isset($_POST['logout'])) {
    include 'destroy.php';
}
if ($_SESSION['status'] != 'Active') {
    header('location:login.php');
}
?>
<?php
$output = '';
if (isset($_POST['refresh'])) {
    $output = '';
}
if (isset($_POST['submit'])) {
    if (isset($_POST['symptom1'])) {
        $number1 = 1;
    } else {
        $number1 = 0;
    }
    if (isset($_POST['symptom2'])) {
        $number2 = 1;
    } else {
        $number2 = 0;
    }
    if (isset($_POST['symptom3'])) {
        $number3 = 1;
    } else {
        $number3 = 0;
    }
    if (isset($_POST['symptom4'])) {
        $number4 = 1;
    } else {
        $number4 = 0;
    }
    if (isset($_POST['symptom5'])) {
        $number5 = 1;
    } else {
        $number5 = 0;
    }
    if (isset($_POST['symptom6'])) {
        $number6 = 1;
    } else {
        $number6 = 0;
    }
    if (isset($_POST['symptom7'])) {
        $number7 = 1;
    } else {
        $number7 = 0;
    }
    if (isset($_POST['symptom8'])) {
        $number8 = 1;
    } else {
        $number8 = 0;
    }
    if (isset($_POST['symptom9'])) {
        $number9 = 1;
    } else {
        $number9 = 0;
    }
    if (isset($_POST['symptom10'])) {
        $number10 = 1;
    } else {
        $number10 = 0;
    }
    if (isset($_POST['symptom11'])) {
        $number11 = 1;
    } else {
        $number11 = 0;
    }
    if (isset($_POST['symptom12'])) {
        $number12 = 1;
    } else {
        $number12 = 0;
    }
    if (isset($_POST['symptom13'])) {
        $number13 = 1;
    } else {
        $number13 = 0;
    }
    if (isset($_POST['symptom14'])) {
        $number14 = 1;
    } else {
        $number14 = 0;
    }
    if (isset($_POST['symptom15'])) {
        $number15 = 1;
    } else {
        $number15 = 0;
    }
    if (isset($_POST['symptom16'])) {
        $number16 = 1;
    } else {
        $number16 = 0;
    }
    if (isset($_POST['symptom17'])) {
        $number17 = 1;
    } else {
        $number17 = 0;
    }
    if (isset($_POST['symptom18'])) {
        $number18 = 1;
    } else {
        $number18 = 0;
    }
    if (isset($_POST['symptom19'])) {
        $number19 = 1;
    } else {
        $number19 = 0;
    }
    if (isset($_POST['symptom20'])) {
        $number20 = 1;
    } else {
        $number20 = 0;
    }
    if (isset($_POST['symptom21'])) {
        $number21 = 1;
    } else {
        $number21 = 0;
    }
    if (isset($_POST['symptom22'])) {
        $number22 = 1;
    } else {
        $number22 = 0;
    }
    if (isset($_POST['symptom23'])) {
        $number23 = 1;
    } else {
        $number23 = 0;
    }
    if (isset($_POST['symptom24'])) {
        $number24 = 1;
    } else {
        $number24 = 0;
    }
    if (isset($_POST['symptom25'])) {
        $number25 = 1;
    } else {
        $number25 = 0;
    }
    if (isset($_POST['symptom26'])) {
        $number26 = 1;
    } else {
        $number26 = 0;
    }
    if (isset($_POST['symptom27'])) {
        $number27 = 1;
    } else {
        $number27 = 0;
    }
    if (isset($_POST['symptom28'])) {
        $number28 = 1;
    } else {
        $number28 = 0;
    }
    if (isset($_POST['symptom29'])) {
        $number29 = 1;
    } else {
        $number29 = 0;
    }
    if (isset($_POST['symptom30'])) {
        $number30 = 1;
    } else {
        $number30 = 0;
    }
    if (isset($_POST['symptom31'])) {
        $number31 = 1;
    } else {
        $number31 = 0;
    }
    if (isset($_POST['symptom32'])) {
        $number32 = 1;
    } else {
        $number32 = 0;
    }
    if (isset($_POST['symptom33'])) {
        $number33 = 1;
    } else {
        $number33 = 0;
    }
    if (isset($_POST['symptom34'])) {
        $number34 = 1;
    } else {
        $number34 = 0;
    }
    if (isset($_POST['symptom35'])) {
        $number35 = 1;
    } else {
        $number35 = 0;
    }
    if (isset($_POST['symptom36'])) {
        $number36 = 1;
    } else {
        $number36 = 0;
    }
    if (isset($_POST['symptom37'])) {
        $number37 = 1;
    } else {
        $number37 = 0;
    }
    if (isset($_POST['symptom38'])) {
        $number38 = 1;
    } else {
        $number38 = 0;
    }
    if (isset($_POST['symptom39'])) {
        $number39 = 1;
    } else {
        $number39 = 0;
    }
    if (isset($_POST['symptom40'])) {
        $number40 = 1;
    } else {
        $number40 = 0;
    }
    if (isset($_POST['symptom41'])) {
        $number41 = 1;
    } else {
        $number41 = 0;
    }
    if (isset($_POST['symptom42'])) {
        $number42 = 1;
    } else {
        $number42 = 0;
    }
    if (isset($_POST['symptom43'])) {
        $number43 = 1;
    } else {
        $number43 = 0;
    } // if(isset($_POST['symptom44'])) // else //  $number44=0;
    //   $number44 = 1;
    if (isset($_POST['symptom45'])) {
        $number45 = 1;
    } else {
        $number45 = 0;
    }
    if (isset($_POST['symptom46'])) {
        $number46 = 1;
    } else {
        $number46 = 0;
    }
    if (isset($_POST['symptom47'])) {
        $number47 = 1;
    } else {
        $number47 = 0;
    }
    if (isset($_POST['symptom48'])) {
        $number48 = 1;
    } else {
        $number48 = 0;
    }
    if (isset($_POST['symptom49'])) {
        $number49 = 1;
    } else {
        $number49 = 0;
    }
    if (isset($_POST['symptom50'])) {
        $number50 = 1;
    } else {
        $number50 = 0;
    }
    if (isset($_POST['symptom51'])) {
        $number51 = 1;
    } else {
        $number51 = 0;
    }
    if (isset($_POST['symptom52'])) {
        $number52 = 1;
    } else {
        $number52 = 0;
    }
    if (isset($_POST['symptom53'])) {
        $number53 = 1;
    } else {
        $number53 = 0;
    }
    if (isset($_POST['symptom54'])) {
        $number54 = 1;
    } else {
        $number54 = 0;
    }
    if (isset($_POST['symptom55'])) {
        $number55 = 1;
    } else {
        $number55 = 0;
    }
    if (isset($_POST['symptom56'])) {
        $number56 = 1;
    } else {
        $number56 = 0;
    }
    if (isset($_POST['symptom57'])) {
        $number57 = 1;
    } else {
        $number57 = 0;
    }
    if (isset($_POST['symptom58'])) {
        $number58 = 1;
    } else {
        $number58 = 0;
    }
    if (isset($_POST['symptom59'])) {
        $number59 = 1;
    } else {
        $number59 = 0;
    }
    if (isset($_POST['symptom60'])) {
        $number60 = 1;
    } else {
        $number60 = 0;
    }
    if (isset($_POST['symptom61'])) {
        $number61 = 1;
    } else {
        $number61 = 0;
    }
    if (isset($_POST['symptom62'])) {
        $number62 = 1;
    } else {
        $number62 = 0;
    }
    if (isset($_POST['symptom63'])) {
        $number63 = 1;
    } else {
        $number63 = 0;
    }
    if (isset($_POST['symptom64'])) {
        $number64 = 1;
    } else {
        $number64 = 0;
    }
    if (isset($_POST['symptom65'])) {
        $number65 = 1;
    } else {
        $number65 = 0;
    }
    if (isset($_POST['symptom66'])) {
        $number66 = 1;
    } else {
        $number66 = 0;
    }
    if (isset($_POST['symptom67'])) {
        $number67 = 1;
    } else {
        $number67 = 0;
    }
    if (isset($_POST['symptom68'])) {
        $number68 = 1;
    } else {
        $number68 = 0;
    }
    if (isset($_POST['symptom69'])) {
        $number69 = 1;
    } else {
        $number69 = 0;
    }
    if (isset($_POST['symptom70'])) {
        $number70 = 1;
    } else {
        $number70 = 0;
    }
    if (isset($_POST['symptom71'])) {
        $number71 = 1;
    } else {
        $number71 = 0;
    }
    if (isset($_POST['symptom72'])) {
        $number72 = 1;
    } else {
        $number72 = 0;
    }
    if (isset($_POST['symptom73'])) {
        $number73 = 1;
    } else {
        $number73 = 0;
    }
    if (isset($_POST['symptom74'])) {
        $number74 = 1;
    } else {
        $number74 = 0;
    }
    if (isset($_POST['symptom75'])) {
        $number75 = 1;
    } else {
        $number75 = 0;
    }
    if (isset($_POST['symptom76'])) {
        $number76 = 1;
    } else {
        $number76 = 0;
    }
    if (isset($_POST['symptom77'])) {
        $number77 = 1;
    } else {
        $number77 = 0;
    }
    if (isset($_POST['symptom78'])) {
        $number78 = 1;
    } else {
        $number78 = 0;
    }
    if (isset($_POST['symptom79'])) {
        $number79 = 1;
    } else {
        $number79 = 0;
    }
    if (isset($_POST['symptom80'])) {
        $number80 = 1;
    } else {
        $number80 = 0;
    }
    if (isset($_POST['symptom81'])) {
        $number81 = 1;
    } else {
        $number81 = 0;
    }
    if (isset($_POST['symptom82'])) {
        $number82 = 1;
    } else {
        $number82 = 0;
    }
    if (isset($_POST['symptom83'])) {
        $number83 = 1;
    } else {
        $number83 = 0;
    }
    if (isset($_POST['symptom84'])) {
        $number84 = 1;
    } else {
        $number84 = 0;
    }
    if (isset($_POST['symptom85'])) {
        $number85 = 1;
    } else {
        $number85 = 0;
    }
    if (isset($_POST['symptom86'])) {
        $number86 = 1;
    } else {
        $number86 = 0;
    }
    if (isset($_POST['symptom87'])) {
        $number87 = 1;
    } else {
        $number87 = 0;
    }
    if (isset($_POST['symptom88'])) {
        $number88 = 1;
    } else {
        $number88 = 0;
    }
    if (isset($_POST['symptom89'])) {
        $number89 = 1;
    } else {
        $number89 = 0;
    }
    if (isset($_POST['symptom90'])) {
        $number90 = 1;
    } else {
        $number90 = 0;
    }
    if (isset($_POST['symptom91'])) {
        $number91 = 1;
    } else {
        $number91 = 0;
    }
    if (isset($_POST['symptom92'])) {
        $number92 = 1;
    } else {
        $number92 = 0;
    }
    if (isset($_POST['symptom93'])) {
        $number93 = 1;
    } else {
        $number93 = 0;
    }
    if (isset($_POST['symptom94'])) {
        $number94 = 1;
    } else {
        $number94 = 0;
    }
    if (isset($_POST['symptom95'])) {
        $number95 = 1;
    } else {
        $number95 = 0;
    }
    if (isset($_POST['symptom96'])) {
        $number96 = 1;
    } else {
        $number96 = 0;
    }
    if (isset($_POST['symptom97'])) {
        $number97 = 1;
    } else {
        $number97 = 0;
    }
    if (isset($_POST['symptom98'])) {
        $number98 = 1;
    } else {
        $number98 = 0;
    }
    if (isset($_POST['symptom99'])) {
        $number99 = 1;
    } else {
        $number99 = 0;
    }
    if (isset($_POST['symptom100'])) {
        $number100 = 1;
    } else {
        $number100 = 0;
    }
    if (isset($_POST['symptom101'])) {
        $number101 = 1;
    } else {
        $number101 = 0;
    }
    if (isset($_POST['symptom102'])) {
        $number102 = 1;
    } else {
        $number102 = 0;
    }
    if (isset($_POST['symptom103'])) {
        $number103 = 1;
    } else {
        $number103 = 0;
    }
    if (isset($_POST['symptom104'])) {
        $number104 = 1;
    } else {
        $number104 = 0;
    }
    if (isset($_POST['symptom105'])) {
        $number105 = 1;
    } else {
        $number105 = 0;
    }
    if (isset($_POST['symptom106'])) {
        $number106 = 1;
    } else {
        $number106 = 0;
    }
    if (isset($_POST['symptom107'])) {
        $number107 = 1;
    } else {
        $number107 = 0;
    }
    if (isset($_POST['symptom108'])) {
        $number108 = 1;
    } else {
        $number108 = 0;
    }
    if (isset($_POST['symptom109'])) {
        $number109 = 1;
    } else {
        $number109 = 0;
    }
    if (isset($_POST['symptom110'])) {
        $number110 = 1;
    } else {
        $number110 = 0;
    }
    if (isset($_POST['symptom111'])) {
        $number111 = 1;
    } else {
        $number111 = 0;
    }
    if (isset($_POST['symptom112'])) {
        $number112 = 1;
    } else {
        $number112 = 0;
    }
    if (isset($_POST['symptom113'])) {
        $number113 = 1;
    } else {
        $number113 = 0;
    }
    if (isset($_POST['symptom114'])) {
        $number114 = 1;
    } else {
        $number114 = 0;
    }
    if (isset($_POST['symptom115'])) {
        $number115 = 1;
    } else {
        $number115 = 0;
    }
    if (isset($_POST['symptom116'])) {
        $number116 = 1;
    } else {
        $number116 = 0;
    }
    if (isset($_POST['symptom117'])) {
        $number117 = 1;
    } else {
        $number117 = 0;
    }
    if (isset($_POST['symptom118'])) {
        $number118 = 1;
    } else {
        $number118 = 0;
    }
    if (isset($_POST['symptom119'])) {
        $number119 = 1;
    } else {
        $number119 = 0;
    }
    if (isset($_POST['symptom120'])) {
        $number120 = 1;
    } else {
        $number120 = 0;
    }
    if (isset($_POST['symptom121'])) {
        $number121 = 1;
    } else {
        $number121 = 0;
    }
    if (isset($_POST['symptom122'])) {
        $number122 = 1;
    } else {
        $number122 = 0;
    }
    if (isset($_POST['symptom123'])) {
        $number123 = 1;
    } else {
        $number123 = 0;
    }
    if (isset($_POST['symptom124'])) {
        $number124 = 1;
    } else {
        $number124 = 0;
    }
    if (isset($_POST['symptom125'])) {
        $number125 = 1;
    } else {
        $number125 = 0;
    }
    if (isset($_POST['symptom126'])) {
        $number126 = 1;
    } else {
        $number126 = 0;
    }
    if (isset($_POST['symptom127'])) {
        $number127 = 1;
    } else {
        $number127 = 0;
    }
    if (isset($_POST['symptom128'])) {
        $number128 = 1;
    } else {
        $number128 = 0;
    }
    if (isset($_POST['symptom129'])) {
        $number129 = 1;
    } else {
        $number129 = 0;
    }
    if (isset($_POST['symptom130'])) {
        $number130 = 1;
    } else {
        $number130 = 0;
    }
    if (isset($_POST['symptom131'])) {
        $number131 = 1;
    } else {
        $number131 = 0;
    }
    if (isset($_POST['symptom132'])) {
        $number132 = 1;
    } else {
        $number132 = 0;
    }
    $number133 = 2;
    $myArray = [
        $number1,
        $number2,
        $number3,
        $number4,
        $number5,
        $number6,
        $number7,
        $number8,
        $number9,
        $number10,
        $number11,
        $number12,
        $number13,
        $number14,
        $number15,
        $number16,
        $number17,
        $number18,
        $number19,
        $number20,
        $number21,
        $number22,
        $number23,
        $number24,
        $number25,
        $number26,
        $number27,
        $number28,
        $number29,
        $number30,
        $number31,
        $number32,
        $number33,
        $number34,
        $number35,
        $number36,
        $number37,
        $number38,
        $number39,
        $number40,
        $number41,
        $number42,
        $number43,
        $number45,
        $number46,
        $number47,
        $number48,
        $number49,
        $number50,
        $number51,
        $number52,
        $number53,
        $number54,
        $number55,
        $number56,
        $number57,
        $number58,
        $number59,
        $number60,
        $number61,
        $number62,
        $number63,
        $number64,
        $number65,
        $number66,
        $number67,
        $number68,
        $number69,
        $number70,
        $number71,
        $number72,
        $number73,
        $number74,
        $number75,
        $number76,
        $number77,
        $number78,
        $number79,
        $number80,
        $number81,
        $number82,
        $number83,
        $number84,
        $number85,
        $number86,
        $number87,
        $number88,
        $number89,
        $number90,
        $number91,
        $number92,
        $number93,
        $number94,
        $number95,
        $number96,
        $number97,
        $number98,
        $number99,
        $number100,
        $number101,
        $number102,
        $number103,
        $number104,
        $number105,
        $number106,
        $number107,
        $number108,
        $number109,
        $number110,
        $number111,
        $number112,
        $number113,
        $number114,
        $number115,
        $number116,
        $number117,
        $number118,
        $number119,
        $number120,
        $number121,
        $number122,
        $number123,
        $number124,
        $number125,
        $number126,
        $number127,
        $number128,
        $number129,
        $number130,
        $number131,
        $number132,
        $number133,
    ];
    $string = '';
    $count = 0;
    foreach ($myArray as $num) {
        if ($count != 131) {
            $string = $string . strval($num) . ',';
        } else {
            $string = $string . strval($num);
        }
        $count++;
    }
    $string2 = [
        'abdominal_pain',
        'abnormal_menstruation',
        'acidity,acute_liver_failure',
        'altered_sensorium',
        'anxiety',
        'back_pain',
        'belly_pain',
        'blackheads',
        'bladder_discomfort',
        'blister',
        'blood_in_sputum',
        'bloody_stool',
        'blurred_and_distorted_vision',
        'breathlessness',
        'brittle_nails',
        'bruising',
        'burning_micturition',
        'chest_pain',
        'chills',
        'cold_hands_and_feets',
        'coma',
        'congestion',
        'constipation',
        'continuous_feel_of_urine',
        'continuous_sneezing',
        'cough',
        'cramps',
        'dark_urine',
        'dehydration',
        'depression',
        'diarrhoea',
        'dischromic _patches',
        'distention_of_abdomen',
        'dizziness',
        'drying_and_tingling_lips',
        'enlarged_thyroid',
        'excessive_hunger',
        'extra_marital_contacts',
        'family_history',
        'fast_heart_rate',
        'fatigue',
        'fluid_overload',
        'foul_smell_of urine',
        'headache',
        'high_fever',
        'hip_joint_pain',
        'history_of_alcohol_consumption',
        'increased_appetite',
        'indigestion',
        'inflammatory_nails',
        'internal_itching',
        'irregular_sugar_level',
        'irritability',
        'irritation_in_anus',
        'itching',
        'joint_pain',
        'knee_pain',
        'lack_of_concentration',
        'lethargy',
        'loss_of_appetite',
        'loss_of_balance',
        'loss_of_smell',
        'malaise',
        'mild_fever',
        'mood_swings',
        'movement_stiffness',
        'mucoid_sputum',
        'muscle_pain',
        'muscle_wasting',
        'muscle_weakness',
        'nausea',
        'neck_pain',
        'nodal_skin_eruptions',
        'obesity',
        'pain_behind_the_eyes',
        'pain_during_bowel_movements',
        'pain_in_anal_region',
        'painful_walking',
        'palpitations',
        'passage_of_gases',
        'patches_in_throat',
        'phlegm',
        'polyuria',
        'prominent_veins_on_calf',
        'puffy_face_and_eyes',
        'pus_filled_pimples',
        'receiving_blood_transfusion',
        'receiving_unsterile_injections',
        'red_sore_around_nose',
        'red_spots_over_body',
        'redness_of_eyes',
        'restlessness',
        'runny_nose',
        'rusty_sputum',
        'scurring',
        'shivering',
        'silver_like_dusting',
        'sinus_pressure',
        'skin_peeling',
        'skin_rash',
        'slurred_speech',
        'small_dents_in_nails',
        'spinning_movements',
        'spotting_ urination',
        'stiff_neck',
        'stomach_bleeding',
        'stomach_pain',
        'sunken_eyes',
        'sweating',
        'swelled_lymph_nodes',
        'swelling_joints',
        'swelling_of_stomach',
        'swollen_blood_vessels',
        'swollen_extremeties',
        'swollen_legs',
        'throat_irritation,toxic_look_(typhos)',
        'ulcers_on_tongue',
        'unsteadiness',
        'visual_disturbances',
        'vomiting',
        'watering_from_eyes',
        'weakness_in_limbs',
        'weakness_of_one_body_side',
        'weight_gain',
        'weight_loss',
        'yellow_crust_ooze',
        'yellow_urine',
        'yellowing_of_eyes',
        'yellowish_skin',
        'prognosis',
        "\n",
    ];
    $c = 0;
    $string3 = '';
    $len = sizeof($string2);
    foreach ($string2 as $s) {
        if ($c == $len - 1 || $c == $len - 2) {
            $string3 = $string3 . $s;
        } else {
            $string3 = $string3 . $s . ',';
        }
        $c++;
    }
    $abc = fopen('Book1.csv', 'w');
    fwrite($abc, $string3);
    fwrite($abc, $string); // echo $string;
    $command = escapeshellcmd('python checkup.py &');
    $output = shell_exec($command);
}
?>
<!DOCTYPE html>
<html lang="en">
<head>

     <title>Disease Predictor</title>

<script src="https://kit.fontawesome.com/b62e73b8ee.js" crossorigin="anonymous"></script>
  <style >
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
	position: relative;
	width: 60%;
	left: 5%;
  background-color: #a5c422;
  text-align: center;
  border-radius: 3px;
	border: 2px solid darkslategrey;
	margin: 5px;
	margin-bottom:3vw;
	padding: 10px;
	transition: background 0.5s,box-shadow 1s;
}
  .class1:hover{
	background:linear-gradient(135deg,black,darkslategrey);
	box-shadow: 0px 0px 10px 4px black;

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


     <link rel="stylesheet" href="css/tooplate-style.css">


</head>
<body id="top" data-spy="scroll" data-target=".navbar-collapse" data-offset="50">


     <section class="preloader">
          <div class="spinner">

               <span class="spinner-rotate"></span>

          </div>
     </section>



     <header>
          <div class="container">
               <div class="row">

                    <div class="col-md-4 col-sm-5">
                         <p>Welcome to Health Care Bot</p>
                    </div>

                    <div class="col-md-8 col-sm-7 text-align-right">
                         <span class="phone-icon"><i class="fa fa-phone"></i> +91-9330554877</span>
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
                         <li><a href="#id1" class="smoothScroll">Contact</a></li>
                        <li><form method="post"><button type="submit"name="logout" class="btn1">LogOut</button></form></li>
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
                              <h2 class="wow fadeInUp" data-wow-delay="0.6s">Welcome to Your <i class="fa fa-h-square"></i>ealth Care Bot</h2>
                              <div class="wow fadeInUp" data-wow-delay="0.8s">
                                   <p>Improve health and care with us</p>

                              </div>
                              <figure class="profile wow fadeInUp" data-wow-delay="1s">
                                   <img src="images/Bot.jpeg" class="img-responsive" alt="">
                                   <figcaption>
                                        <h3>Dr.Bot</h3>
                                        <p>General Principal</p>
                                   </figcaption>
                              </figure>
                         </div>
                    </div>

               </div>
          </div>
     </section>



    <section id="predictor" data-stellar-background-ratio="3">
        <div class="container">
               <div class="row">

                    <div class="col-md-6 col-sm-6">
                         <img src="images/Bot.jpeg" class="img-responsive" alt="">
                    </div>

                    <div class="col-md-6 col-sm-6">
                         <!-- CONTACT FORM HERE -->
                         <form id="appointment-form" role="form" method="post" action="#">

                              <!-- SECTION TITLE -->
                              <div class="section-title wow fadeInUp" data-wow-delay="0.4s">
                                   <h2>What are your symptoms?</h2>
                              </div>

                              <div class="wow fadeInUp" data-wow-delay="0.8s">


                                   <div class="col-md-12 col-sm-12">

                                        <div>
                                          <h3 class="class1">  <label for="select"><a href="new-detail.php">Click me to Select Symptoms <i class="fas fa-stethoscope"></i></i></a></label></h3>
                                        </div>
                                   </div>

                                   <div class="col-md-12 col-sm-12">

                                            <h3>Result:</h3><h2><?php if (
                                                $output != ''
                                            ) {
                                                echo $output;
                                            } ?></h2>

                                        <button type="submit" class="btn1" id="cf-submit" name="refresh">Refresh</button>
                                   </div>
                              </div>
                        </form>
                    </div>

               </div>
          </div>
    </section>
     <!-- MAKE AN APPOINTMENT -->
     <section id="appointment" data-stellar-background-ratio="3">
          <div class="container">
               <div class="row">

                    <div class="col-md-6 col-sm-6">
                         <!-- CONTACT FORM HERE -->
                         <form id="appointment-form" role="form" method="post" action="MAILTO:devgoyal2028@gmail.com" enctype="text/plain">

                              <!-- SECTION TITLE -->

                              <div class="section-title wow fadeInUp" data-wow-delay="0.4s">
                                   <h2>Make an appointment</h2>
                              </div>


                              <div class="wow fadeInUp" data-wow-delay="0.8s">
                                   <div class="col-md-6 col-sm-6">

                                        <label for="name">Name</label>

                                        <input type="text" class="form-control" id="name" name="Name" placeholder="Full Name"/>

                                   </div>

                                   <div class="col-md-6 col-sm-6">
                                        <label for="email">Email</label>
                                        <input type="email" class="form-control" id="email" name="email" placeholder="Your Email"/>
                                   </div>

                                   <div class="col-md-6 col-sm-6">
                                        <label for="date">Select Date</label>
                                        <input type="date" name="date" value="" class="form-control">
                                   </div>

                                   <div class="col-md-6 col-sm-6">
                                        <label for="select">Select Department</label>
                                        <select class="form-control">
                                             <option>General Health</option>
                                             <option>Cardiology</option>
                                             <option>Dental</option>
                                             <option>Medical Research</option>
                                        </select>
                                   </div>

                                   <div class="col-md-12 col-sm-12">
                                        <label for="telephone">Phone Number</label>
                                        <input type="tel" class="form-control" id="phone" name="phone" placeholder="Phone">
                                        <label for="Message">Additional Message</label>
                                        <textarea class="form-control" rows="5" id="message" name="message" placeholder="Message"></textarea>

                                       <button type="submit" class="form-control" id="cf-submit" >Submit Button</button>
                                   </div>


                             </div>
                        </form>
                    </div>

                   <div class="col-md-6 col-sm-6">
                         <img src="images/appointment-image.jpg" class="img-responsive" alt="">
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
