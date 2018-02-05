FILE="$1"
HEADER="$2"
NEWNAME="$3"

tail -n +2 $FILE | sed 's/"\([^"]*\)"/\1\t/g' | sed "s/\t,/\t/g" > tmp.tsv
cat $HEADER tmp.tsv > $NEWNAME
rm tmp.tsv
