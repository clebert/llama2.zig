pub fn argmax(vector: []f32) usize {
    var max_index: usize = 0;
    var max_element: f32 = vector[max_index];

    for (1..vector.len) |index| {
        const element = vector[index];

        if (element > max_element) {
            max_index = index;
            max_element = element;
        }
    }

    return max_index;
}
