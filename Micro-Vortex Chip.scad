//////////////////////////////////////////////////////////////
// Micro-Vortex Trap Chip — OpenSCAD (units: mm)
// עדן + GPT | F5 Preview, F6 Render
//////////////////////////////////////////////////////////////

//======================== VIEW / MODE ======================//
presenter_mode   = true;     // true=עבה ונוח למצגת, false=ריאליסטי
show_exploded    = false;    // תצוגה מפוצלת לשכבות
transparent_view = true;

$fn = 128;

//======================== BASE SIZE ========================//
slide_size    = [75, 25];    // גודל סלייד סטנדרטי
margin_xy     = 3;           // שוליים סביב התעלה

//======================== THICKNESSES ======================//
t_glass_real  = 0.17;        // #1.5H
t_glass_show  = 1.2;
t_glass       = presenter_mode ? t_glass_show : t_glass_real;

t_pdms_real   = 2.0;         // עובי PDMS טיפוסי
t_pdms_show   = 3.5;
t_pdms        = presenter_mode ? t_pdms_show : t_pdms_real;

//======================== CHANNEL GEOMETRY =================//
// הערוץ הראשי "יושב" על הזכוכית (חיסור מתוך תחתית ה-PDMS)
ch_w_real     = 0.15;        // רוחב הערוץ (ריאליסטי: 150 µm)
ch_w_show     = 1.0;         // לנראות במסך
ch_w          = presenter_mode ? ch_w_show : ch_w_real;

ch_h_real     = 0.08;        // גובה הערוץ (80 µm)
ch_h_show     = 0.8;
ch_h          = presenter_mode ? ch_h_show : ch_h_real;

// אורך מקטע פעיל (בין כניסה ליציאה)
active_len    = slide_size[0] - 2*margin_xy;

// “כיסי-מערבולת” (Side Chambers)
trap_R_show   = 1.6;         // רדיוס כיס (מצגת)
trap_R_real   = 0.6;         // ~600 µm (ייצור)
trap_R        = presenter_mode ? trap_R_show : trap_R_real;

trap_pitch    = presenter_mode ? 5.0 : 2.0;   // מרחק בין כיסים עוקבים (מרכז-מרכז)
n_traps       = floor((active_len - 10) / trap_pitch); // חישוב כמות בקירוב
trap_offset_y = 0.6*(presenter_mode ? 2.0 : 0.8);      // הסטה מציר הערוץ (מרכז→צד)

// צוואר כניסה לכיס (nozzle) — יוצר זרימה סיבובית
nozzle_w      = ch_w*0.6;    // מצר את הכניסה
nozzle_len    = presenter_mode ? 0.8 : 0.25;

//======================== PORTS / RESERVOIRS ===============//
port_diam     = presenter_mode ? 3.0 : 1.8;   // חור לכניסה/יציאה מלמעלה
res_diam      = presenter_mode ? 6.0 : 3.5;   // מאגר עילי רדוד
res_depth     = presenter_mode ? 1.6 : 0.8;

corner_R      = 2.0;         // רדיוס פינות ל-PDMS

//======================== COLORS ===========================//
col_glass = transparent_view ? [0.25,0.8,1.0,0.25] : [0.7,0.9,1.0,1];
col_pdms  = transparent_view ? [0.95,0.6,0.95,0.25] : [1,0.7,1,1];

//======================== HELPERS ==========================//
sx = slide_size[0];
sy = slide_size[1];

// מסלול הערוץ (מרכז) על הזכוכית
y_center = sy/2;

// נקודות התחלה/סיום (ליד השוליים)
x0 = margin_xy;
x1 = sx - margin_xy;

//======================== MODULES ==========================//
module rounded_plate(x,y,z,r){
  minkowski(){
    cube([x-2*r,y-2*r,z], center=false);
    cylinder(h=0.001,r=r);
  }
}

module glass(){
  color(col_glass) cube([sx,sy,t_glass], center=false);
}

module pdms_block(){
  color(col_pdms)
    translate([0,0,t_glass])
      rounded_plate(sx, sy, t_pdms, corner_R);
}

// חיסור הערוץ הראשי
module main_channel(){
  translate([x0, y_center - ch_w/2, t_glass])
    cube([active_len, ch_w, ch_h+0.01], center=false);
}

// כיס מערבולת אחד (צד ימין/שמאל)
module one_trap(cx, side){
  // side = +1 למעלה, -1 למטה (יחס לציר הערוץ)
  // כיס עגול בגובה הערוץ
  translate([cx, y_center + side*trap_offset_y, t_glass])
    linear_extrude(height=ch_h+0.01)
      circle(r=trap_R);

  // צוואר כניסה צר בין הערוץ לכיס
  translate([cx - nozzle_len/2, y_center + side*(trap_offset_y - nozzle_w/2), t_glass])
    cube([nozzle_len, nozzle_w, ch_h+0.01], center=false);
}

// מערך כיסי-מערבולת משני צדי הערוץ
module traps_array(){
  // התחלה מעט אחרי הכניסה ועד לפני היציאה
  start_x = x0 + 5;
  for (i=[0:n_traps-1]){
    cx = start_x + i*trap_pitch;
    one_trap(cx, +1);
    one_trap(cx, -1);
  }
}

// כניסה/יציאה + מאגרים עיליים
module ports(){
  // כניסה
  translate([x0, y_center, t_glass + t_pdms - res_depth])
    cylinder(h=res_depth+0.02, d=res_diam);
  translate([x0, y_center, t_glass])
    cylinder(h=t_pdms+0.02, d=port_diam);

  // יציאה
  translate([x1, y_center, t_glass + t_pdms - res_depth])
    cylinder(h=res_depth+0.02, d=res_diam);
  translate([x1, y_center, t_glass])
    cylinder(h=t_pdms+0.02, d=port_diam);

  // חיבור המאגר אל הערוץ (משפך רדוד)
  translate([x0, y_center - ch_w, t_glass])
    cube([2.0, ch_w*3, ch_h+0.01], center=false);
  translate([x1-2.0, y_center - ch_w, t_glass])
    cube([2.0, ch_w*3, ch_h+0.01], center=false);
}

// בלוק PDMS עם כל החיסורים (ערוץ+כיסים+פורטים)
module pdms_layer(){
  difference(){
    pdms_block();
    main_channel();
    traps_array();
    ports();
  }
}

//======================== EXPLODED VIEW ====================//
module exploded(){
  glass();
  translate([0,0,t_glass + 1.5]) pdms_block();
  // “צל” של הערוץ והכיסים להמחשה
  color([0,0,0,0.15]){
    translate([0,0,t_glass + 1.5])
      difference(){
        rounded_plate(sx, sy, t_pdms, corner_R);
        translate([0,0,1.0]){ main_channel(); traps_array(); ports(); }
      }
  }
}

//======================== ASSEMBLY =========================//
if (show_exploded){
  exploded();
} else {
  glass();
  pdms_layer();
}

/*************  הערות שימוש קצרות  **************************
1) presenter_mode=true מציג קנה-מידה “מצגת”. לבדיקת מידות אמיתיות:
   set presenter_mode=false  (הערוץ ~150µm רוחב, 80µm גובה; כיס ~600µm).
2) יצוא STL לשכבת PDMS: השאירו show_exploded=false → F6 Render → Export STL.
3) חיבור ניסויי:
   - כניסה/יציאה: שני חורים עיליים (port_diam). ניתן להדביק Luer stubs.
   - זרימה: 0.5–5 µL/min להתחלה. מערבולת נוצרת בכיסים ו”לוכדת” תאים.
*************************************************************/
