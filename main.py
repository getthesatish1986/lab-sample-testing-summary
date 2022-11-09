from flask import Flask, render_template, request
import pandas as pd
import pickle
import statsmodels


app = Flask(__name__,template_folder='template')

@app.route('/')
@app.route('/home')
def home():
    return render_template('formpage.html')

@app.route('/confirm',methods=["POST"])
def confirm():
    if request.method == 'POST':
         a = request.form.get('test_booking_time')
         b = request.form.get('collection_time')
         c = request.form.get('cutoff_time')
         d = request.form.get('Agent_Location_KM')
         e = request.form.get('Time_Taken_To_Reach_Patient_MM')
         f = request.form.get('Time_For_Sample_Collection_MM')
         g = request.form.get('Lab_Location_KM')
         h = request.form.get('Time_Taken_To_Reach_Lab_MM')
         i = 0
         if (request.form.getlist('gender'))=="Male":
            i = 1
         j=0
         if (request.form.getlist('test_name'))=="CBC":
            j = 1
         k=0
         if (request.form.getlist('test_name'))=="Complete Urinalysis":
            k = 1
         l = 0
         if (request.form.getlist('test_name')) == "Fasting blood sugar":
             l = 1
         m = 0
         if (request.form.getlist('test_name')) == "H1N1":
             m = 1
         n = 0
         if (request.form.getlist('test_name')) == "HbA1c":
             n = 1
         o = 0
         if (request.form.getlist('test_name')) == "Lipid Profile":
             o = 1
         p = 0
         if (request.form.getlist('test_name')) == "RTPCR":
             p = 1
         q = 0
         if (request.form.getlist('test_name')) == "TSH":
             q = 1
         r = 0
         if (request.form.getlist('test_name')) == "Vitamin D-25Hydroxy":
             r = 1
         s = 0
         if (request.form.getlist('sample')) == "Swab":
             s = 1
         t = 0
         if (request.form.getlist('sample')) == "Urine":
             t = 1
         u = 0
         if (request.form.getlist('sample')) == "blood":
             u = 1
         v = 0
         if (request.form.getlist('storage')) == "Normal":
             v = 1
         w = 0
         if (request.form.getlist('cutoff_schedule')) == "Sample by 5PM":
             w = 1
         x = 0
         if (request.form.getlist('Traffic')) == "Low Traffic":
             x = 1
         y = 0
         if (request.form.getlist('Traffic')) == "Medium Traffic":
             y = 1

    file_name = 'my_file.pkl'
    model = pickle.load(open(file_name,'rb'))
    #data = pd.DataFrame([int(a), 13, 17, 7, 14, 10, 13, 26, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])
    #data = pd.DataFrame([int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0])
    data = pd.DataFrame([float(a),float(b), float(c),int(d), int(e), int(f),int(g),int(h),int(i),int(j),int(k),int(l),int(m),int(n),int(o),int(p),int(q),int(r),int(s),int(t),int(u),int(v),int(w),int(x),int(y)])
    fi = data.transpose()
    fi.columns = ['Test_Booking_Time_HH_MM', 'Scheduled_Sample_Collection_Time_HH_MM', 'Cut_off_time_HH_MM',
                  'Agent_Location_KM',
                  'Time_Taken_To_Reach_Patient_MM', 'Time_For_Sample_Collection_MM', 'Lab_Location_KM',
                  'Time_Taken_To_Reach_Lab_MM',
                  'Patient_Gender_Male', 'Test_Name_CBC', 'Test_Name_Complete_Urinalysis',
                  'Test_Name_Fasting_blood_sugar', 'Test_Name_H1N1', 'Test_Name_HbA1c',
                  'Test_Name_Lipid_Profile', 'Test_Name_RTPCR', 'Test_Name_TSH', 'Test_Name_Vitamin_D_25Hydroxy',
                  'Sample_Swab', 'Sample_Urine',
                  'Sample_blood', 'Way_Of_Storage_Of_Sample_Normal', 'Cut_off_Schedule_Sample_by_5pm',
                  'Traffic_Conditions_Low_Traffic',
                  'Traffic_Conditions_Medium_Traffic']
    val = model.predict(fi)[0]
    #print((model.predict(fi))[0])
    if round(val) ==1:
        return render_template('formpage.html',xx = "Reached on Time: yes")
    else:
        return render_template('formpage.html',xx = "Reached on Time: no")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
