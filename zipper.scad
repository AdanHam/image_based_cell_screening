//////////////////////////////////////////////////////////////
// Option C (C1) — Thick Presentation View (Single FOV)
// Units: mm | F5 Preview, F6 Render
//////////////////////////////////////////////////////////////

//======================== VIEW MODE ========================//
show_mode = "pre";      // "pre"  (לפני משיכה) או "post" (אחרי משיכה)

// הצגה עבה (true) או ריאלית-אופטית (false)
presenter_mode = true;

//======================== BASE PARAMS (PHYSICAL) ===========//
// Glass sizes
glass_bot_size   = [60, 25];     // bottom slide/platform
glass_top_size   = [18, 18];     // top coverslip

// Physical (real) thicknesses
t_glass_real     = 2;            // לצורך הצגה (הוגדל)
frame_thk_real   = 2;            // לצורך הצגה (הוגדל)
zip_thk_real     = 1;            // פער לפני מגע (הוגדל להצגה)
bead_diam_real   = 1;            // חרוזים — הוגדל להצגה

// Window & ports
win_size         = [14, 14];
wall_min         = 5;
port_diam        = 2.0;
port_offset_x    = 10;

//======================== ZIPPER (parametric) ==============//
zip_w            = 5;            // רוחב ה-zipper (מ"מ)
zip_L            = 80;           // **אורך ה-zipper** (מ"מ) — עכשיו ארוך יותר
zip_round        = 1.5;          // קצה מוביל מעוגל

//======================== VISUAL THICKENING =================//
t_glass_vis      = 2;
frame_thk_vis    = 2;
zip_thk_vis      = 1;
bead_diam_vis    = 0.25;

//======================== COLOR & CAMERA ====================//
col_bot_glass    = [0.6,0.8,1.0,0.18];
col_top_glass    = [0.6,0.8,1.0,0.35];
col_frame        = [0.08,0.08,0.08,0.95];
col_zipper       = [1.0,0.6,0.2,0.95];
col_bead         = [0.4,0.4,0.4,0.9];

$vpt = [0,0,10];
$vpr = [63,0,35];
$vpd = 120;
$fn  = 64;

//======================== DERIVED ==========================//
frame_size   = [60, 25];
valid_wx     = (frame_size[0] - win_size[0]) >= 2*wall_min;
valid_wy     = (frame_size[1] - win_size[1]) >= 2*wall_min;

// איזה עובי להשתמש בהצגה
t_glass      = presenter_mode ? t_glass_vis   : t_glass_real;
frame_thk    = presenter_mode ? frame_thk_vis : frame_thk_real;
zip_thk_draw = presenter_mode ? zip_thk_vis   : zip_thk_real;
bead_diam    = presenter_mode ? bead_diam_vis : bead_diam_real;

// Z references (פערים לפי "פיזי" כדי לשמר היגיון)
z_bot     = 0;
z_frame   = z_bot + t_glass;
z_preTop  = z_frame + max(zip_thk_real, 0.01);   // gap before pull
z_postTop = z_frame + max(bead_diam_real, 0);    // after pull (beads) או 0

//======================== MODULES ==========================//
module glass_xy(sizeXY, t, col) {
  color(col)
  translate([-sizeXY[0]/2, -sizeXY[1]/2, 0])
    cube([sizeXY[0], sizeXY[1], t], center=false);
}

module frame_adhesive() {
  outer = [frame_size[0], frame_size[1]];
  inner = [win_size[0],   win_size[1]];
  if (valid_wx && valid_wy) {
    difference() {
      color(col_frame)
      translate([-outer[0]/2, -outer[1]/2, 0])
        cube([outer[0], outer[1], frame_thk], center=false);

      // חלון פנימי
      translate([-inner[0]/2, -inner[1]/2, -0.05])
        cube([inner[0], inner[1], frame_thk+0.1], center=false);

      // חריץ צד ל-zipper — מותאם לרוחב ה-zipper
      slit_len = inner[1] + 4;
      translate([ -outer[0]/2 + 0.3, 0, frame_thk/2 ])
        rotate([0,90,0])
          cube([0.6, slit_len, zip_w + 0.5], center=true);

      // פורטים
      for (sx=[-1,1]) {
        translate([sx*port_offset_x, 0, frame_thk/2])
          rotate([90,0,0])
            cylinder(h=outer[1], r=port_diam/2, center=true);
      }
    }
  } else {
    color(col_frame)
    translate([-outer[0]/2, -outer[1]/2, 0])
      cube([outer[0], outer[1], frame_thk], center=false);
  }
}

module zipper_strip() {
  // Kapton/PSA strip — אורך ורוחב פרמטריים
  lead = zip_round;
  body = zip_L - lead;
  color(col_zipper)
  union() {
    translate([-zip_L/2 + lead, -zip_w/2, 0])
      cube([body, zip_w, zip_thk_draw], center=false);
    translate([-zip_L/2 + lead, 0, zip_thk_draw/2])
      rotate([90,0,0])
        cylinder(h=zip_w, r=lead, center=true);
  }
}

module beads_triangle(zlvl=0) {
  pts = [
    [ 9, 0],
    [ -9*cos(60),  9*sin(60)],
    [ -9*cos(60), -9*sin(60)]
  ];
  for (i=[0:2]) {
    translate([pts[i][0], pts[i][1], zlvl])
      color(col_bead)
        sphere(r = bead_diam/2);
  }
}

//======================== SCENES ===========================//
// Pre-contact (zipper in, gap = zip_thk_real)
module scene_pre() {
  translate([0,0,z_bot])   glass_xy(glass_bot_size, t_glass, col_bot_glass);
  translate([0,0,z_frame]) frame_adhesive();
  // zipper בין הזכוכיות — ארוך יותר, יוצא שמאלה יפה
  translate([-(frame_size[0]/2 - 0.6), 0, z_frame + max(zip_thk_real, 0.01)/2])
    zipper_strip();
  translate([0,0,z_preTop]) glass_xy(glass_top_size, t_glass, col_top_glass);
}

// Post-pull (zipper out; top rests on beads or contact)
module scene_post() {
  translate([0,0,z_bot])   glass_xy(glass_bot_size, t_glass, col_bot_glass);
  translate([0,0,z_frame]) frame_adhesive();
  translate([0,0,z_bot + t_glass + 0.01]) beads_triangle();
  translate([0,0,z_postTop]) glass_xy(glass_top_size, t_glass, col_top_glass);
}

//======================== MAIN =============================//
if (show_mode == "pre") {
  scene_pre();
} else { // "post"
  scene_post();
}
