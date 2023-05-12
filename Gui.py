import tkinter as tk
import Gender_model

def predict_gender():
    userid = int(entry_userid.get())
    age = int(entry_age.get())
    dob_day = int(entry_dob_day.get())
    dob_month = int(entry_dob_month.get())
    dob_year = int(entry_dob_year.get())
    tenure = int(entry_tenure.get())
    friend_count = int(entry_friend_count.get())
    friendships_initiated = int(entry_friendships_initiated.get())
    likes = int(entry_likes.get())
    likes_received = int(entry_likes_received.get())
    mobile_likes = int(entry_mobile_likes.get())
    mobile_likes_received = int(entry_mobile_likes_received.get())
    www_likes = int(entry_www_likes.get())
    www_likes_received = int(entry_www_likes_received.get())
    
    # Perform gender prediction using the model
    predicted_gender = [userid, age, dob_day, dob_month, dob_year, tenure, friend_count,
                        friendships_initiated, likes, likes_received, mobile_likes,
                        mobile_likes_received, www_likes, www_likes_received]
    gender_prediction = Gender_model.best_model.predict([predicted_gender])
    disp_string = "This prediction has an accuracy of: " + "{:.2%}".format(Gender_model.model_accuracy)

    if gender_prediction == 0:
        result_string = disp_string + '\n' + "0 - You are predicted to be Female."
    else:
        result_string = disp_string + '\n' + "1 - You are predicted to be male."
    
    # Update the result label with the predicted gender
    label_result.config(text="Predicted Gender: " + result_string)


# Create the main window
window = tk.Tk()
window.title("Gender Prediction")

# Create labels
label_userid = tk.Label(window, text="User ID:")
label_age = tk.Label(window, text="Age:")
label_dob_day = tk.Label(window, text="DOB Day:")
label_dob_month = tk.Label(window, text="DOB Month:")
label_dob_year = tk.Label(window, text="DOB Year:")
label_tenure = tk.Label(window, text="Tenure:")
label_friend_count = tk.Label(window, text="Friend Count:")
label_friendships_initiated = tk.Label(window, text="Friendships Initiated:")
label_likes = tk.Label(window, text="Likes:")
label_likes_received = tk.Label(window, text="Likes Received:")
label_mobile_likes = tk.Label(window, text="Mobile Likes:")
label_mobile_likes_received = tk.Label(window, text="Mobile Likes Received:")
label_www_likes = tk.Label(window, text="WWW Likes:")
label_www_likes_received = tk.Label(window, text="WWW Likes Received:")
label_result = tk.Label(window, text="Predicted Gender:")

# Create entry fields
entry_userid = tk.Entry(window)
entry_age = tk.Entry(window)
entry_dob_day = tk.Entry(window)
entry_dob_month = tk.Entry(window)
entry_dob_year = tk.Entry(window)
entry_tenure = tk.Entry(window)
entry_friend_count = tk.Entry(window)
entry_friendships_initiated = tk.Entry(window)
entry_likes = tk.Entry(window)
entry_likes_received = tk.Entry(window)
entry_mobile_likes = tk.Entry(window)
entry_mobile_likes_received = tk.Entry(window)
entry_www_likes = tk.Entry(window)
entry_www_likes_received = tk.Entry(window)

# Create predict button
button_predict = tk.Button(window, text="Predict", command=predict_gender)

# Place the components on the window
label_userid.pack()
entry_userid.pack()
label_age.pack()
entry_age.pack()
label_dob_day.pack()
entry_dob_day.pack()
label_dob_month.pack()
entry_dob_month.pack()
label_dob_year.pack()
entry_dob_year.pack()
label_tenure.pack()
entry_tenure.pack()
label_friend_count.pack()
entry_friend_count.pack()
label_friendships_initiated.pack()
entry_friendships_initiated.pack()
label_likes.pack()
entry_likes.pack()
label_likes_received.pack()
entry_likes_received.pack()
label_mobile_likes.pack()
entry_mobile_likes.pack()
label_mobile_likes_received.pack()
entry_mobile_likes_received.pack()
label_www_likes.pack()
entry_www_likes.pack()
label_www_likes_received.pack()
entry_www_likes_received.pack()
button_predict.pack()
label_result.pack()

# Start the GUI main loop
window.mainloop()
