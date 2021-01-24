import streamlit as st

from src.preparation import *
from src.model import *

from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)

dl_score = 0.0
al_score = 0.0


def general_config():
    lr = st.number_input('Learning Rate:', value=1e-3, min_value=0.0, max_value=2.0, format='%f')
    st.write("Learning rate: ", lr)

    epochs = st.number_input('Epochs:', value=10, min_value=10, max_value=100, format='%d')
    st.write("Epochs: ", epochs)

    batch_size = st.number_input('Batch size:', value=128, min_value=32, max_value=1024, step=32, format='%d')
    st.write("Batch size: ", batch_size)

    return lr, epochs, batch_size

    
def al_config():
    n_initial = st.number_input('N initial:', value=100, min_value=10, max_value=1000, format='%d')
    st.write("N initial: ", n_initial)

    n_queries = st.number_input('N queries:', value=10, min_value=10, format='%d')
    st.write("N queries: ", n_queries)

    query_strategy = st.selectbox("Query stratege", ("uncertainty_sampling", "entropy_sampling"))
    st.write('Selected:', query_strategy)
    # client_mode = st.radio("Client Mode",(False, True))

    return n_initial, n_queries, query_strategy

@st.cache
def prepare_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2021)

    return X_train, X_test, y_train, y_test


def main():
    global al_score, dl_score
    st.title("MNIST - An Ng.")
    
    X_train, X_test, y_train, y_test = prepare_data()
    st.markdown('**Shape**')
    st.write('\nTraining set :',X_train.shape, "\nTest set :",X_test.shape)

    X_train, y_train = preprocess_data(X_train, y_train.reshape(-1,1))
    X_test, y_test = preprocess_data(X_test, y_test.reshape(-1,1))

    # general_config()
    radio_btn = st.radio(
        "Approach",
        ("Deep Learning", "Active Learning")
    )

    if radio_btn == "Deep Learning":
        col1, col2 = st.beta_columns([1,2])

        #params
        with col1:
            dl_expander = st.beta_expander("Params", expanded=True)
            with dl_expander:
                lr, epochs, batch_size = general_config()
        #display
        
        with col2:
            if st.button("Train"):
                #training
                with st.beta_container():
                    model = PassiveLearner(X_train, y_train, X_test, y_test, epochs, batch_size, lr)
                    with st.spinner('Training...'):
                        model.train()
                    st.balloons()
                    st.success("Train Successfully")
                    
                    dl_score = model.evaluate(X_test, y_test)
                    st.write("Accuracy of Deep learning: ",dl_score)
                    
    else:
        col1, col2 = st.beta_columns([1,2])
        
        #params
        with col1:
            al_expander = st.beta_expander("Params", expanded=True)
            with al_expander:
                lr, epochs, batch_size = general_config()
                n_initial, n_queries, query_strategy = al_config()
                
                if query_strategy == 'uncertainty_sampling':
                    query_strategy = uncertainty_sampling
                else:
                    query_strategy = entropy_sampling
        #display
        with col2:
            if st.button("Train"):
                #training
                with st.beta_container():
                    model = CustomAcitveLearner(X_train, y_train, X_test, y_test, epochs, batch_size, lr, n_initial, n_queries, query_strategy)
                    
                    with st.spinner('Training...'):
                        model.train()
                    
                    st.balloons()
                    st.success("Train Successfully")

                    al_score = model.evaluate(X_test, y_test)
                    st.write("Accuracy of Active learning: ",al_score)
    

def select_box(idx):
    number_opt = st.selectbox("Is digit", [0,1,2,3,4,5,6,7,8,9])
    number_opt = np.array([int(number_opt)])
    # number_opt = st.number_input("is digit", min_value=0, max_value=9, format="%d", key=idx)
    return number_opt

if __name__ == "__main__":
    main()
    
    # result_holder = st.empty()
    
    # with result_holder:
    #     try:
    #         st.write('\n- Deep learning: ', dl_score, "\n- Active learning:",al_score)
    #     except:
    #         try:
    #             st.write('\n- Deep learning: ', dl_score, "\n- Active learning:",0.0)
    #         except:
    #             st.write('\n- Deep learning: ', 0.0, "\n- Active learning:",al_score)
