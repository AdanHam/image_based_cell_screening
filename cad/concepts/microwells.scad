//////////////////////////////////////////////////////////////
// Micro-Well Forest (Microwell Cassette) — OpenSCAD
// Units: mm | F5 Preview, F6 Render
// Author: Adan + GPT
//////////////////////////////////////////////////////////////

//======================== VIEW / MODE ======================//
show_exploded      = false;    // להדגמה: פיצול בין שכבות
presenter_mode     = true;     // true= עבה ונוח למצגת, false= ריאליסטי-אופטי
transparent_parts  = true;     // זכוכית/PDMS שקופים ויפים

$fn = 128;  // עידון עיגולים

//======================== PHYSICAL PARAMS ==================//
// מידות הבסיס (גודל סלייד סטנדרטי 75x25 או כיסוי זכוכית קטן יותר)
base_size      = [75, 25];         // X,Y mm של הבסיס
edge_margin    = 3;                // שוליים חופשיים בלי באריות

// זכוכית תחתונה (להדמיה ברזולוציה גבוהה)
t_glass_real   = 0.17;             // #1.5H ≈ 0.17 mm
t_glass_show   = 1.2;              // עובי "מצגת" כדי לראות בעין
t_glass        = presenter_mode ? t_glass_show : t_glass_real;

// שכבת PDMS נושאת באריות
t_pdms_real    = 1.5;              // טיפוסי לצ'יפים (1–2 mm)
t_pdms_show    = 3.0;              // עבה יותר לנראות
t_pdms         = presenter_mode ? t_pdms_show : t_pdms_real;

// באריות: קוטר, עומק, מרווח (pitch), וסידור
well_diam      = 40;               // קוטר בארית (20–50 מומלץ; כאן 40 µm? לא—זה ב-mm! לכן נבחר 0.04?)
// חשוב: אנו עובדים ב-mm. אם תרצי מיקרונים, חלקי ב-1000.
// ניתן להגדיר מיקרו-גרסה בהמשך. כאן נעשה גרסה "מאקרו" להדגמה,
// וגרסה אמיתית במציאות תיוצר בפבריקציה אחרת.
// לצורך הדמיה על מסך, נשתמש בבאריות גדולות יותר: 1.5 mm קוטר.
well_diam      = 1.5;             // קוטר בארית (mm) — ערך דמואי ונוח לראות
well_depth     = min(t_pdms*0.9, 1.2); // עומק בארית (לא חודר את כל ה-PDMS)
well_pitch     = 3.0;             // מרחק מרכז-מרכז בין באריות
nx             = 18;              // מספר באריות בכיוון X
ny             = 6;               // מספר באריות בכיוון Y

// כיסי כניסה/שטיפה אופציונליים (חריצים רדודים מעל שורות קצה)
add_rinse_grooves = true;
groove_width      = 0.6;          // רוחב חריץ
groove_depth      = 0.4;          // עומק חריץ
groove_offset     = 0.8;          // מרחק מהשפה הפנימית של שורת הבאריות

// פינות מעוגלות עדינות לשכבת PDMS
corner_radius   = 2.0;

// חורי יישור/עיגון אופציונליים בשוליים
add_alignment_holes = true;
hole_diam          = 2.0;

// צבעים (RGBA)
col_glass   = transparent_parts ? [0.25, 0.8, 1.0, 0.25] : [0.7,0.9,1.0,1];
col_pdms    = transparent_parts ? [0.95, 0.6, 0.95, 0.25] : [1,0.7,1,1];
col_wells   = [0.1, 0.1, 0.1, 0.2];  // רק להצללה פנימית קלה (לא חובה)

//======================== DERIVED ==========================//
bx = base_size[0];
by = base_size[1];

usable_x = bx - 2*edge_margin;
usable_y = by - 2*edge_margin;

// חשב את מרכז מערך הבאריות כך שישב סימטרי
array_w = (nx-1)*well_pitch;
array_h = (ny-1)*well_pitch;
array_origin = [ (bx - array_w)/2, (by - array_h)/2 ];

//======================== MODULES ==========================//

// בלוק פינות מעוגלות
module rounded_plate(x, y, z, r) {
  minkowski() {
    cube([x - 2*r, y - 2*r, z], center=false);
    cylinder(h=0.0001, r=r);
  }
}

// בסיס זכוכית
module glass_base() {
  color(col_glass)
    translate([0,0,0])
      cube([bx, by, t_glass], center=false);
}

// שכבת PDMS עם באריות מחוסרות
module pdms_layer() {
  difference() {
    // גוף PDMS
    color(col_pdms)
      translate([0,0,t_glass]) 
        rounded_plate(bx, by, t_pdms, corner_radius);

    // באריות (חיסור גלילי)
    wells_cylinders();
    
    // חריצי שטיפה אופציונליים
    if (add_rinse_grooves) grooves();

    // חורי יישור אופציונליים
    if (add_alignment_holes) alignment_holes();
  }
}

// יצירת כל הבאריות כמערך גלילים מחסרים
module wells_cylinders() {
  for (i=[0:nx-1])
    for (j=[0:ny-1]) {
      cx = array_origin[0] + i*well_pitch;
      cy = array_origin[1] + j*well_pitch;
      // מרכז הדופן העליונה של ה-PDMS
      translate([cx, cy, t_glass + t_pdms - well_depth])
        cylinder(h=well_depth + 0.2, d=well_diam); // +0.2 כדי לוודא חיסור מלא
    }
}

// חריצי שטיפה לאורך השורות העליונה/תחתונה
module grooves() {
  // חריץ מעל שורת הבאריות העליונה
  top_y  = array_origin[1] + (ny-1)*well_pitch + groove_offset;
  bot_y  = array_origin[1] - groove_offset;

  groove_len = (nx-1)*well_pitch + well_diam;
  start_x    = array_origin[0] - well_diam/2;
  
  // חריץ עליון
  translate([start_x, top_y - groove_width/2, t_glass + t_pdms - groove_depth])
    cube([groove_len, groove_width, groove_depth + 0.2], center=false);
  // חריץ תחתון
  translate([start_x, bot_y - groove_width/2, t_glass + t_pdms - groove_depth])
    cube([groove_len, groove_width, groove_depth + 0.2], center=false);
}

// חורי יישור בשוליים
module alignment_holes() {
  offset = 6;
  z0 = t_glass; // נתחיל לחסר מה-PDMS כלפי מעלה
  // שמאל-עליון
  translate([offset, by-offset, z0])
    cylinder(h=t_pdms+0.2, d=hole_diam);
  // ימין-עליון
  translate([bx-offset, by-offset, z0])
    cylinder(h=t_pdms+0.2, d=hole_diam);
  // שמאל-תחתון
  translate([offset, offset, z0])
    cylinder(h=t_pdms+0.2, d=hole_diam);
  // ימין-תחתון
  translate([bx-offset, offset, z0])
    cylinder(h=t_pdms+0.2, d=hole_diam);
}

// תצוגה מפוצלת אופציונלית
module exploded_view() {
  // זכוכית
  translate([0,0,0]) glass_base();
  // רווח אוויר להדגמה
  translate([0,0,t_glass + 0.8]) 
    color([0,0,0,0]) cube([bx, by, 0.001], center=false);
  // PDMS
  translate([0,0,t_glass + 1.6]) pdms_layer_block_only(); // גוף בלבד
  // “רוחות” של הבאריות להמחשה (חצי שקוף)
  wells_shadow();
}

// גוף PDMS בלבד (ללא חיסורים) — לשימוש במצב מפוצל
module pdms_layer_block_only() {
  color(col_pdms)
    rounded_plate(bx, by, t_pdms, corner_radius);
}

// הדמיה חלשה של נפחי הבאריות (לא חובה)
module wells_shadow() {
  color(col_wells)
    for (i=[0:nx-1])
      for (j=[0:ny-1]) {
        cx = array_origin[0] + i*well_pitch;
        cy = array_origin[1] + j*well_pitch;
        translate([cx, cy, t_glass + t_pdms - well_depth])
          cylinder(h=well_depth, d=well_diam);
      }
}

//======================== ASSEMBLY =========================//
if (show_exploded) {
  exploded_view();
} else {
  // הרכבה מלאה
  glass_base();
  pdms_layer();
}

//======================== NOTES ============================//
// 1) הערכים כאן "ידידותיים לעין" במסך. בייצור אמיתי (מיקרונים):
//    - well_diam טיפוסי: 20–50 µm  => הגדרה ב-mm: 0.02–0.05
//    - well_pitch טיפוסי: 40–80 µm => 0.04–0.08 mm
//    שימי לב ש-OpenSCAD מציג מ"מ, אז תצטרכי להתאים סולם.
// 2) ל-STL של שכבת ה-PDMS בלבד: כבי את חיסור הבאריות (להדמיה),
//    או עשי Render (F6) ואז File→Export→STL לשכבה.
// 3) אפשר להוסיף "מכסה" עליון (lid) עם חרירי כניסה/יציאה ולחבר צינוריות.
//
