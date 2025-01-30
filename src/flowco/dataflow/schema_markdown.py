from flowco.dataflow.extended_type import *


def main():
    print("### Example 1: Simple Primitive Types\n")

    # Sample 1: Simple IntType
    int_type = IntType(description="A unique identifier for a user.")
    extended_type_int = ExtendedType(the_type=int_type, description="User ID Type.")
    schema_int = extended_type_int.type_schema()
    markdown_int = schema_to_text(schema_int)
    print((markdown_int))
    print("\n---\n")

    print("### Example 2: Optional String Type\n")

    # Sample 2: Optional StrType
    str_type = StrType(description="The user's middle name.")
    optional_str_type = OptionalType(
        wrapped_type=str_type, description="User's middle name which is optional."
    )
    extended_type_optional_str = ExtendedType(
        the_type=optional_str_type, description="Optional Middle Name Type."
    )
    schema_optional_str = extended_type_optional_str.type_schema()
    markdown_optional_str = schema_to_text(schema_optional_str)
    print((markdown_optional_str))
    print("\n---\n")

    print("### Example 3: List of Floats\n")

    # Sample 3: List of Floats
    float_type = FloatType(description="A single sensor reading.")
    list_of_floats = ListType(
        element_type=float_type,
        length=None,  # Arbitrary length
        description="A list of sensor readings.",
    )
    extended_type_list_floats = ExtendedType(
        the_type=list_of_floats, description="Sensor Readings List Type."
    )
    schema_list_floats = extended_type_list_floats.type_schema()
    markdown_list_floats = schema_to_text(schema_list_floats)
    print((markdown_list_floats))
    print("\n---\n")

    print("### Example 4: Dictionary with String Keys and Boolean Values\n")

    # Sample 4: Dict[str, bool]
    bool_type = BoolType(description="Feature enabled status.")
    dict_type = DictType(
        key_type=StrType(description="Feature name."),
        value_type=bool_type,
        description="Mapping of feature names to their enabled status.",
    )
    extended_type_dict = ExtendedType(
        the_type=dict_type, description="Feature Flags Dictionary Type."
    )
    schema_dict = extended_type_dict.type_schema()
    markdown_dict = schema_to_text(schema_dict)
    print((markdown_dict))
    print("\n---\n")

    print("### Example 5: Record with Various Fields\n")

    # Sample 5: Record with various fields
    id_type = IntType(description="Unique user identifier.")
    name_type = StrType(description="Full name of the user.")
    email_type = OptionalType(
        wrapped_type=StrType(description="User's email address."),
        description="Email is optional.",
    )
    age_type = OptionalType(
        wrapped_type=IntType(description="Age of the user."),
        description="Age is optional.",
    )

    key_id = KeyType(key="id", type=id_type, description="User ID.")
    key_name = KeyType(key="name", type=name_type, description="User's full name.")
    key_email = KeyType(
        key="email", type=email_type, description="User's email address."
    )
    key_age = KeyType(key="age", type=age_type, description="User's age.")

    record = RecordType(
        name="UserProfile",
        items=[key_id, key_name, key_email, key_age],
        description="A user's profile information.",
    )
    extended_type_record = ExtendedType(
        the_type=record, description="User Profile Record Type."
    )
    schema_record = extended_type_record.type_schema()
    markdown_record = schema_to_text(schema_record)
    print((markdown_record))
    print("\n---\n")

    print("### Example 6: Pandas DataFrame with Specific Column Types\n")

    # Sample 6: Pandas DataFrame
    # Define column types
    col_id = KeyType(
        key="id",
        type=IntType(description="User's unique identifier."),
        description="User ID column.",
    )
    col_name = KeyType(
        key="name", type=StrType(description="User's name."), description="Name column."
    )
    col_active = KeyType(
        key="active",
        type=BoolType(description="User's active status."),
        description="Active status column.",
    )

    dataframe_type = PDDataFrameType(
        columns=[col_id, col_name, col_active],
        description="DataFrame containing user information.",
    )
    extended_type_dataframe = ExtendedType(
        the_type=dataframe_type, description="User DataFrame Type."
    )
    schema_dataframe = extended_type_dataframe.type_schema()
    markdown_dataframe = schema_to_text(schema_dataframe)
    print((markdown_dataframe))
    print("\n---\n")

    print("### Example 7: Nested Structures (List of Records)\n")

    # Sample 7: Nested List of Records
    product_id_type = IntType(description="Product identifier.")
    product_name_type = StrType(description="Name of the product.")
    product_price_type = FloatType(description="Price of the product.")

    product_key_id = KeyType(key="id", type=product_id_type, description="Product ID.")
    product_key_name = KeyType(
        key="name", type=product_name_type, description="Product name."
    )
    product_key_price = KeyType(
        key="price", type=product_price_type, description="Product price."
    )

    product_record = RecordType(
        name="Product",
        items=[product_key_id, product_key_name, product_key_price],
        description="A single product.",
    )

    list_of_products = ListType(
        element_type=product_record, length=None, description="A list of products."
    )

    extended_type_list_products = ExtendedType(
        the_type=list_of_products, description="List of Product Records."
    )
    schema_list_products = extended_type_list_products.type_schema()
    markdown_list_products = schema_to_text(schema_list_products)
    print((markdown_list_products))
    print("\n---\n")

    print("### Example 8: NumPy ndarray of Integers\n")

    # Sample 8: NumPy ndarray of Integers
    ndarray_type = NumpyNdarrayType(
        element_type=IntType(description="Pixel intensity value."),
        length=1024,  # Example length
        description="NumPy array of pixel intensities.",
    )
    extended_type_ndarray = ExtendedType(
        the_type=ndarray_type, description="Pixel Intensities ndarray Type."
    )
    schema_ndarray = extended_type_ndarray.type_schema()
    markdown_ndarray = schema_to_text(schema_ndarray)
    print((markdown_ndarray))
    print("\n---\n")

    print("### Example 9: Pandas Series with Optional Float\n")

    # Sample 9: Pandas Series with Optional Float
    optional_float_type = OptionalType(
        wrapped_type=FloatType(description="Measurement value."),
        description="Measurement can be a float or None.",
    )

    series_type = PDSeriesType(
        element_type=optional_float_type, description="Series of measurement values."
    )

    extended_type_series = ExtendedType(
        the_type=series_type, description="Measurements Series Type."
    )
    schema_series = extended_type_series.type_schema()
    markdown_series = schema_to_text(schema_series)
    print((markdown_series))
    print("\n---\n")

    print("### Example 10: Complex Nested Structure\n")

    # Sample 10: Complex Nested Structure
    # Define types for order
    order_id_type = IntType(description="Order identifier.")
    customer_name_type = StrType(description="Name of the customer.")
    customer_email_type = OptionalType(
        wrapped_type=StrType(description="Email of the customer."),
        description="Customer email is optional.",
    )

    # Define customer Record
    customer_record = RecordType(
        name="Customer",
        items=[
            KeyType(key="id", type=order_id_type, description="Customer ID."),
            KeyType(key="name", type=customer_name_type, description="Customer name."),
            KeyType(
                key="email", type=customer_email_type, description="Customer email."
            ),
        ],
        description="Customer information.",
    )

    # Define product Record
    product_record_order = RecordType(
        name="Product",
        items=[
            KeyType(
                key="id",
                type=IntType(description="Product ID."),
                description="Product ID.",
            ),
            KeyType(
                key="name",
                type=StrType(description="Product name."),
                description="Product name.",
            ),
            KeyType(
                key="price",
                type=FloatType(description="Product price."),
                description="Product price.",
            ),
        ],
        description="Product information.",
    )

    # Define order Record
    order_record = RecordType(
        name="Order",
        items=[
            KeyType(key="order_id", type=order_id_type, description="Unique order ID."),
            KeyType(
                key="customer",
                type=customer_record,
                description="Customer details.",
            ),
            KeyType(
                key="items",
                type=ListType(
                    element_type=product_record_order,
                    length=None,
                    description="List of products in the order.",
                ),
                description="Ordered items.",
            ),
        ],
        description="Order details.",
    )

    extended_type_order = ExtendedType(
        the_type=order_record, description="Complete Order Type."
    )
    schema_order = extended_type_order.type_schema()
    markdown_order = schema_to_text(schema_order)
    print((markdown_order))
    print("\n---\n")


if __name__ == "__main__":
    main()
