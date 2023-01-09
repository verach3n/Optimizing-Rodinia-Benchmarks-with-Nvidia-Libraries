for i in {102400..409600..10240}
do
    nvprof ./gaussian_managedm $i  
done