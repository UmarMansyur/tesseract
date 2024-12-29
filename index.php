<?php

$a = [1, 2, 3];

unset($a[0]);
$a[0] = 2;
echo $a[0];
print_r($a);